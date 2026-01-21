"""
jTrans_instr Model for Pretraining

Standard BERT architecture with instruction-level embeddings:
- Token embeddings: word_embeddings (like BERT)
- Position embeddings: position_embeddings (like BERT, token-level positions 0-511)
- Instruction embeddings: NEW embedding layer (instruction indices 0-200)

Final embedding = token_embedding + position_embedding + instruction_embedding

Special handling for instr_addr tokens:
- instr_addr_0 to instr_addr_200 directly use instruction embeddings 0-200
- This creates a semantic link: when model sees "jmp instr_addr_5", the instr_addr_5 
  token uses the SAME embedding as instruction #5 in the instruction_embeddings layer
- This helps the model understand that jump targets reference actual instructions

Example:
  If instruction #5 is "mov eax, ebx", and we have a "jmp instr_addr_5":
  - The tokens [mov, eax, ebx] all have instruction_ids=5 → use instruction_embeddings[5]
  - The token "instr_addr_5" directly uses instruction_embeddings[5]
  - Model learns: jump target connects to the actual instruction representation

This differs from baseline jTrans which uses position_embeddings = word_embeddings trick.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings


class InstrBertEmbeddings(BertEmbeddings):
    """
    BERT embeddings with instruction-level embeddings.
    
    Constructs embeddings from three components:
    - word_embeddings: Token embeddings (what token is it?)
    - position_embeddings: Token-level positions (where in sequence? 0-511)
    - instruction_embeddings: Instruction index for each token (which instruction? 0-200)
    
    Maps tokens to instructions: multiple tokens belong to the same instruction.
    
    Special handling for instr_addr_{i} tokens:
    - instr_addr_0 to instr_addr_200 directly use instruction embeddings 0-200
    - This creates semantic link between jump addresses and instruction representations
    
    Final: embeddings = word + position + instruction
    (NO token_type_embeddings - we don't need segment IDs)
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Add instruction embeddings (instruction indices 0-200)
        self.max_instructions = getattr(config, 'max_instructions', 201)
        self.instruction_embeddings = nn.Embedding(
            self.max_instructions, 
            config.hidden_size
        )
        
        # Register buffer for instruction IDs
        self.register_buffer(
            "instruction_ids",
            torch.arange(self.max_instructions).expand((1, -1)),
            persistent=False
        )
        
        # Store vocab to token ID mapping for instr_addr tokens
        # These will be set after tokenizer is loaded
        self.instr_addr_token_ids = None  # Will be set to list of token IDs for instr_addr_0 to instr_addr_200
    
    def set_instr_addr_token_ids(self, tokenizer):
        """
        Set the token IDs for instr_addr_0 to instr_addr_200 from tokenizer.
        Call this after loading the tokenizer.
        
        Args:
            tokenizer: The tokenizer with vocab containing instr_addr tokens
        """
        self.instr_addr_token_ids = []
        for i in range(self.max_instructions):
            token = f"instr_addr_{i}"
            if token in tokenizer.vocab:
                token_id = tokenizer.vocab[token]
                self.instr_addr_token_ids.append(token_id)
            else:
                # If token not found, append -1 (won't match anything)
                self.instr_addr_token_ids.append(-1)
    
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        instruction_ids=None,  # Maps each token to its instruction index
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        """
        Args:
            input_ids: [batch, seq_len] - Token IDs
            position_ids: [batch, seq_len] - Token positions (0, 1, 2, ...)
            instruction_ids: [batch, seq_len] - Instruction index for each token
                Example: [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, ...]
                         |instr0| |i1| |instr2   | |i3|
            inputs_embeds: [batch, seq_len, hidden] - Pre-computed embeddings (optional)
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Generate position IDs if not provided (0, 1, 2, 3, ...)
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Get token embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            
            # Special handling: replace instr_addr_{i} token embeddings with instruction embeddings
            # This creates direct link between jump addresses and instruction representations
            if self.instr_addr_token_ids is not None and input_ids is not None:
                for instr_idx, token_id in enumerate(self.instr_addr_token_ids):
                    if instr_idx >= self.max_instructions:
                        break
                    # Find positions where this instr_addr token appears
                    mask = (input_ids == token_id)
                    if mask.any():
                        # Replace with instruction embedding
                        instr_emb = self.instruction_embeddings.weight[instr_idx]
                        inputs_embeds[mask] = instr_emb
        
        # Get position embeddings (token-level)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Start with token + position
        embeddings = inputs_embeds + position_embeddings
        
        # Add instruction embeddings (maps tokens to instructions)
        if instruction_ids is not None:
            instruction_embeddings = self.instruction_embeddings(instruction_ids)
            embeddings = embeddings + instruction_embeddings
        
        # Layer norm and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class InstrBertModel(BertPreTrainedModel):
    """
    BERT model with instruction-level embeddings.
    
    This uses standard BERT architecture but adds instruction embeddings
    to help the model learn instruction-level structure.
    """
    
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        
        # Use custom embeddings with instruction support
        from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
        
        self.embeddings = InstrBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        
        # Initialize weights
        self.post_init()
    
    def set_instr_addr_token_ids(self, tokenizer):
        """
        Set the token IDs for instr_addr tokens from tokenizer.
        This enables special handling where instr_addr_{i} uses instruction embedding i.
        
        Args:
            tokenizer: The tokenizer with vocab containing instr_addr tokens
        """
        self.embeddings.set_instr_addr_token_ids(tokenizer)
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        instruction_ids=None,  # Maps tokens to instruction indices
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Get embeddings with instruction information
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            instruction_ids=instruction_ids,  # Pass instruction IDs (token->instruction mapping)
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class InstrPretrainingModel(nn.Module):
    """
    jTrans_instr pretraining model with MLM + JTP tasks.
    
    Uses instruction-level embeddings to help the model understand
    instruction boundaries and structure.
    """
    
    def __init__(self, bert_model, vocab_size, max_instructions=201):
        super().__init__()
        self.bert = bert_model
        self.vocab_size = vocab_size
        self.max_instructions = max_instructions
        
        hidden_size = bert_model.config.hidden_size
        
        # MLM head: Predict token vocabulary
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # JTP head: Predict jump target instruction index (0 to max_instructions-1)
        self.jtp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, max_instructions)
        )
    
    def set_instr_addr_token_ids(self, tokenizer):
        """
        Set the token IDs for instr_addr tokens from tokenizer.
        This enables special handling where instr_addr_{i} uses instruction embedding i.
        
        Args:
            tokenizer: The tokenizer with vocab containing instr_addr tokens
        """
        self.bert.set_instr_addr_token_ids(tokenizer)
    
    def forward(self, input_ids, attention_mask, instruction_ids=None):
        """
        Forward pass for jTrans_instr pretraining.
        
        Args:
            input_ids: [batch_size, seq_len] - Token IDs
            attention_mask: [batch_size, seq_len] - Attention mask
            instruction_ids: [batch_size, seq_len] - Instruction index for each token
                Example: [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, ...]
                         Tokens at positions 0,1,2 belong to instruction 0
                         Tokens at positions 3,4 belong to instruction 1
                         etc.
            
        Returns:
            mlm_logits: [batch_size, seq_len, vocab_size]
            jtp_logits: [batch_size, seq_len, max_instructions]
        """
        # Get BERT output with instruction embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            instruction_ids=instruction_ids  # Pass token->instruction mapping
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # MLM prediction
        mlm_logits = self.mlm_head(sequence_output)
        
        # JTP prediction (for instr_addr tokens)
        jtp_logits = self.jtp_head(sequence_output)
        
        return mlm_logits, jtp_logits


def create_instr_model(
    vocab_size,
    max_instructions=201,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
):
    """
    Create jTrans_instr model with instruction-level embeddings.
    
    Args:
        vocab_size: Size of token vocabulary
        max_instructions: Maximum number of instructions (default: 201 for 0-200)
        ... other BERT config parameters
    
    Returns:
        InstrPretrainingModel instance
    """
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=2,
    )
    
    # Add instruction config
    config.max_instructions = max_instructions
    
    # Create BERT model with instruction embeddings
    bert_model = InstrBertModel(config, add_pooling_layer=False)
    
    # Create pretraining model
    model = InstrPretrainingModel(
        bert_model=bert_model,
        vocab_size=vocab_size,
        max_instructions=max_instructions
    )
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing jTrans_instr model...")
    
    vocab_size = 3000
    max_instructions = 201
    batch_size = 2
    seq_len = 128
    
    model = create_instr_model(
        vocab_size=vocab_size,
        max_instructions=max_instructions
    )
    
    # Create a dummy tokenizer with instr_addr tokens
    class DummyTokenizer:
        def __init__(self):
            self.vocab = {}
            # Add special tokens
            self.vocab['[PAD]'] = 0
            self.vocab['[UNK]'] = 1
            self.vocab['[CLS]'] = 2
            self.vocab['[SEP]'] = 3
            self.vocab['[MASK]'] = 4
            # Add instr_addr tokens
            idx = 5
            for i in range(201):
                self.vocab[f'instr_addr_{i}'] = idx
                idx += 1
            # Add some assembly tokens
            self.vocab['mov'] = idx
            self.vocab['add'] = idx + 1
            self.vocab['jmp'] = idx + 2
    
    tokenizer = DummyTokenizer()
    
    # Set instr_addr token IDs so instr_addr_{i} uses instruction embedding i
    model.set_instr_addr_token_ids(tokenizer)
    print("Set instr_addr token IDs for special handling")
    
    # Create dummy inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Add some instr_addr tokens to test special handling
    # Put instr_addr_5 at position 10 - it should use instruction embedding 5
    input_ids[0, 10] = tokenizer.vocab['instr_addr_5']
    # Put instr_addr_10 at position 20 - it should use instruction embedding 10
    input_ids[0, 20] = tokenizer.vocab['instr_addr_10']
    
    # Create instruction IDs: each token maps to an instruction
    # Example: [0,0,0,1,1,2,2,2,2,3,3,...] means:
    #   tokens 0-2 are in instruction 0
    #   tokens 3-4 are in instruction 1
    #   tokens 5-8 are in instruction 2, etc.
    instruction_ids = torch.randint(0, max_instructions, (batch_size, seq_len))
    
    # Forward pass
    mlm_logits, jtp_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        instruction_ids=instruction_ids
    )
    
    print(f"✓ Model created successfully!")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  MLM logits shape: {mlm_logits.shape}")  # [batch, seq_len, vocab_size]
    print(f"  JTP logits shape: {jtp_logits.shape}")  # [batch, seq_len, max_instructions]
    print(f"\n✓ Instruction embeddings added!")
    print(f"  Token embeddings: word_embeddings")
    print(f"  Position embeddings: position_embeddings (token-level 0-511)")
    print(f"  Instruction embeddings: instruction_embeddings (instruction-level 0-200)")
