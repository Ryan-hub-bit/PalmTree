import idc
import idautils
import idaapi
import pickle
import sys
import os

# Add script directory to Python path so util module can be found
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

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
SAVEROOT = os.environ.get('SAVEROOT', '/data/kun/jtrans/instr/extract')
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
            func_name = idc.get_func_name(func)
            asm_list = self.get_asm(func)
            rawbytes_list = self.get_rawbytes(func)
            cfg = self.get_cfg(func)
            bai_feature = self.get_binai_feature(func)
            yield (func_name, func, asm_list, rawbytes_list, cfg, bai_feature)

if __name__ == '__main__':
    import os
    from collections import defaultdict

    assert os.path.exists(DATAROOT), f"DATAROOT does not exist: {DATAROOT}"
    assert os.path.exists(SAVEROOT), f"SAVEROOT does not exist: {SAVEROOT}"

    binary_abs_path = idc.get_input_file_path()
    binary_name = os.path.basename(binary_abs_path).replace('.strip', '')
    
    print(f"[*] Processing binary: {binary_name}")
    print(f"[*] Input file: {binary_abs_path}")
    
    # Find the original unstripped binary
    unstrip_path = None
    for root, dirs, files in os.walk(DATAROOT):
        if binary_name in files:
            unstrip_path = os.path.join(root, binary_name)
            break
    
    if unstrip_path is None:
        print(f"[ERROR] Could not find unstripped binary: {binary_name}")
        idc.qexit(1)
    
    print(f"[*] Found unstripped binary: {unstrip_path}")
    
    try:
        bd = BinaryData(unstrip_path)
        data_list = defaultdict(dict)
        
        func_count = 0
        for func_name, func, asm_list, rawbytes_list, cfg, bai_feature in bd.extract_all():
            data_list[func_name]['func'] = func
            data_list[func_name]['asm'] = asm_list
            data_list[func_name]['raw'] = rawbytes_list
            data_list[func_name]['cfg'] = cfg
            data_list[func_name]['bai'] = bai_feature
            func_count += 1
        
        # Save pickle file
        output_name = binary_name + '_extract.pkl'
        output_path = os.path.join(SAVEROOT, output_name)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data_list, f)
        
        print(f"[SUCCESS] Extracted {func_count} functions")
        print(f"[SUCCESS] Saved to: {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Exception during extraction: {e}")
        import traceback
        traceback.print_exc()
        idc.qexit(1)
    
    idc.qexit(0)
