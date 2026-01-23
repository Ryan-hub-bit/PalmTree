import idc
import idautils
import idaapi
import pickle
import sys

# Add site-packages to Python path so IDA can find installed packages
sys.path.insert(0, '/home/kun/anaconda3/lib/python3.12/site-packages')

# Try to import optional dependencies
try:
    import binaryai
    # Check if binaryai.ida module exists
    if hasattr(binaryai, 'ida'):
        HAS_BINARYAI = True
    else:
        HAS_BINARYAI = False
        print("[WARNING] binaryai found but binaryai.ida not available")
except ImportError:
    HAS_BINARYAI = False
    binaryai = None
    print("[WARNING] binaryai not available - BinaryAI features will be None")

import networkx as nx
from util.base import Binarybase

# Allow overriding save/data roots via environment variables
import os
SAVEROOT = os.environ.get('SAVEROOT', '/data/kun/jtrans_instr/extract')
DATAROOT = os.environ.get('DATAROOT', './dataset')

print(f"[INFO] SAVEROOT={SAVEROOT}")
print(f"[INFO] DATAROOT={DATAROOT}")

class BinaryData(Binarybase):
    def __init__(self, unstrip_path):
        super(BinaryData, self).__init__(unstrip_path)
        self.fix_up()
    
    def fix_up(self):
        for addr in self.addr2name:
            # incase some functions' instructions are not recognized by IDA
            idc.create_insn(addr)  
            idc.add_func(addr) 

    def get_asm(self, func):
        instGenerator = idautils.FuncItems(func)
        asm_list = []
        for inst in instGenerator:
            asm_list.append(idc.GetDisasm(inst))
        return asm_list

    def get_rawbytes(self, func):
        instGenerator = idautils.FuncItems(func)
        rawbytes_list = b""
        for inst in instGenerator:
            rawbytes_list += idc.get_bytes(inst, idc.get_item_size(inst))
        return rawbytes_list

    def get_cfg(self, func):

        def get_attr(block, func_addr_set):
            asm,raw=[],b""
            curr_addr = block.start_ea
            if curr_addr not in func_addr_set:
                return -1
            while curr_addr <= block.end_ea:
                asm.append(idc.GetDisasm(curr_addr))
                raw+=idc.get_bytes(curr_addr, idc.get_item_size(curr_addr))
                curr_addr = idc.next_head(curr_addr, block.end_ea)
            return asm, raw

        nx_graph = nx.DiGraph()
        flowchart = idaapi.FlowChart(idaapi.get_func(func), flags=idaapi.FC_PREDS)
        func_addr_set = set([addr for addr in idautils.FuncItems(func)])
        for block in flowchart:
            # Make sure all nodes are added (including edge-less nodes)
            attr = get_attr(block, func_addr_set)
            if attr == -1:
                continue
            nx_graph.add_node(block.start_ea, asm=attr[0], raw=attr[1])
            for pred in block.preds():
                if pred.start_ea not in func_addr_set:
                    continue
                nx_graph.add_edge(pred.start_ea, block.start_ea)
            for succ in block.succs():
                if succ.start_ea not in func_addr_set:
                    continue
                nx_graph.add_edge(block.start_ea, succ.start_ea)
        return nx_graph  

    def get_binai_feature(self, func):
        if HAS_BINARYAI and binaryai is not None:
            return binaryai.ida.get_func_feature(func)
        else:
            return None

    def extract_all(self):
        for func in idautils.Functions():
            if idc.get_segm_name(func) in ['.plt','extern','.init','.fini']:
                continue
            func_name=idc.get_func_name(func)
            self.func_data[func_name]={
                'asm':self.get_asm(func),
                'cfg':self.get_cfg(func),
                'rawbytes':self.get_rawbytes(func),
                'binai_feature': self.get_binai_feature(func)
            }

if __name__ == '__main__':
    print("[*] Starting IDA Pro extraction for instruction-level representation...")
    idaapi.auto_wait()
    
    input_path = idc.get_input_file_path()
    print(f"[*] Processing binary: {input_path}")
    
    binary = BinaryData(input_path)
    binary.extract_all()
    
    # Save to pickle file
    binary_name = os.path.basename(input_path)
    output_file = os.path.join(SAVEROOT, f'{binary_name}_extract.pkl')
    
    print(f"[*] Saving extracted data to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(binary.func_data, f)
    
    print(f"[*] Extraction complete! Saved {len(binary.func_data)} functions")
    idc.qexit(0)
