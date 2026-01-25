"""
Symbol Abstraction Module for AddressAware Data Generation

This module provides utilities to abstract variable names while preserving
high-value symbols like library functions and standard structure fields.
"""

import re

# ============================================================================
# Standard Library Functions - PRESERVE THESE
# ============================================================================
STDLIB_FUNCTIONS = {
    # Memory management
    'malloc', 'free', 'calloc', 'realloc', 'alloca',
    # String operations
    'strlen', 'strcpy', 'strncpy', 'strcmp', 'strncmp', 'strcat', 'strchr', 'strstr',
    'memcpy', 'memmove', 'memset', 'memcmp',
    # I/O operations
    'printf', 'fprintf', 'sprintf', 'snprintf', 'scanf', 'fscanf',
    'fopen', 'fclose', 'fread', 'fwrite', 'fseek', 'ftell',
    'puts', 'gets', 'fgets', 'fputs', 'fputc', 'fgetc',
    # System calls
    'exit', 'abort', 'atexit', 'system',
    # Threading
    'pthread_create', 'pthread_join', 'pthread_mutex_lock', 'pthread_mutex_unlock',
    # Error handling
    'perror', 'strerror', '__errno_location', '__assert_fail',
    # Math
    'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'pow', 'fabs',
    # Misc
    'qsort', 'bsearch', 'rand', 'srand', 'time',
}

# C++ standard library (mangled names will have _Z prefix)
STDLIB_CPP_PATTERNS = [
    r'_Z.*operator.*',  # C++ operators
    r'_Z.*allocator.*', # std::allocator
    r'_Z.*basic_string.*',  # std::string
    r'_Z.*vector.*',    # std::vector
    r'_Z.*iostream.*',  # std::cout, std::cin
]

# ============================================================================
# Standard Structure Fields (POSIX/Linux) - PRESERVE THESE
# ============================================================================
STANDARD_STRUCT_FIELDS = {
    # struct stat
    'st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_uid', 'st_gid',
    'st_rdev', 'st_size', 'st_blksize', 'st_blocks',
    'st_atime', 'st_mtime', 'st_ctime', 'st_atim', 'st_mtim', 'st_ctim',
    
    # struct timeval / timespec
    'tv_sec', 'tv_nsec', 'tv_usec',
    
    # struct tm
    'tm_sec', 'tm_min', 'tm_hour', 'tm_mday', 'tm_mon', 'tm_year',
    'tm_wday', 'tm_yday', 'tm_isdst', 'tm_gmtoff', 'tm_zone',
    
    # struct sockaddr / sockaddr_in
    'sa_family', 'sa_data', 's_addr', 'sin_port', 'sin_addr',
    
    # struct addrinfo
    'ai_flags', 'ai_family', 'ai_socktype', 'ai_protocol',
    'ai_addrlen', 'ai_addr', 'ai_canonname', 'ai_next',
    
    # struct iovec
    'iov_base', 'iov_len',
    
    # struct msghdr
    'msg_name', 'msg_namelen', 'msg_iov', 'msg_iovlen',
    'msg_control', 'msg_controllen', 'msg_flags',
    
    # struct rlimit
    'rlim_cur', 'rlim_max',
    
    # struct sigaction
    'sa_handler', 'sa_mask', 'sa_flags', '__sigaction_handler',
    
    # struct passwd
    'pw_name', 'pw_passwd', 'pw_uid', 'pw_gid', 'pw_gecos',
    'pw_dir', 'pw_shell',
    
    # struct termios
    'c_iflag', 'c_oflag', 'c_cflag', 'c_lflag', 'c_line',
    'c_cc', 'c_ispeed', 'c_ospeed',
    
    # errno values
    'errno', 'errcode', 'errnum',
    
    # file system
    'f_bsize', 'f_blocks', 'f_bfree', 'f_bavail', 'f_files', 'f_flag',
    
    # process/thread IDs
    'pid', 'tid', 'uid', 'gid', 'euid', 'egid', 'ruid', 'rgid', 'suid', 'sgid',
}

# ============================================================================
# Variable Name Patterns - ABSTRACT THESE
# ============================================================================
LOCAL_VAR_PATTERNS = [
    # Common local variable names
    r'^[a-z][a-z0-9_]{0,15}$',  # Simple lowercase names (code, speed, how, time0)
    r'^tmp[0-9]*$',              # tmp, tmp1, tmp2
    r'^temp[0-9]*$',             # temp, temp1
    r'^var[0-9]*$',              # var, var1, var2
    r'^result[0-9]*$',           # result, result1
    r'^ret[0-9]*$',              # ret, ret1
    r'^val[0-9]*$',              # val, val1
    r'^i[0-9]*$', r'^j[0-9]*$', r'^k[0-9]*$',  # Loop counters
    r'^ptr[0-9]*$',              # ptr, ptr1
    r'^buf[0-9]*$',              # buf, buf1
    r'^len[0-9]*$',              # len, len1
]

GLOBAL_VAR_PATTERNS = [
    # IDA default naming
    r'^byte_[0-9A-F]+$',         # byte_12345678
    r'^word_[0-9A-F]+$',         # word_12345678
    r'^dword_[0-9A-F]+$',        # dword_12345678
    r'^qword_[0-9A-F]+$',        # qword_12345678
    r'^unk_[0-9A-F]+$',          # unk_12345678
    r'^off_[0-9A-F]+$',          # off_12345678
    r'^stru_[0-9A-F]+$',         # stru_12345678
]


def should_preserve_symbol(symbol_name: str) -> bool:
    """
    Determine if a symbol should be preserved (not abstracted).
    
    Preserve:
    - Library function names (especially with _ptr suffix)
    - Standard structure fields
    - Well-known system symbols
    
    Returns:
        True if symbol should be preserved, False if it should be abstracted
    """
    # Always preserve library function pointers (.xxx_ptr)
    if symbol_name.startswith('.') and symbol_name.endswith('_ptr'):
        return True
    
    # Preserve standard library functions
    # Remove _ptr suffix for checking
    clean_name = symbol_name.replace('.', '').replace('_ptr', '')
    if clean_name in STDLIB_FUNCTIONS:
        return True
    
    # Preserve C++ mangled stdlib names
    for pattern in STDLIB_CPP_PATTERNS:
        if re.match(pattern, clean_name):
            return True
    
    # Preserve standard structure fields
    if symbol_name in STANDARD_STRUCT_FIELDS:
        return True
    
    return False


def should_abstract_as_local_var(symbol_name: str) -> bool:
    """Check if symbol matches local variable pattern."""
    if not symbol_name or len(symbol_name) > 20:
        return False
    
    for pattern in LOCAL_VAR_PATTERNS:
        if re.match(pattern, symbol_name):
            return True
    return False


def should_abstract_as_global_var(symbol_name: str) -> bool:
    """Check if symbol matches global variable pattern (IDA defaults)."""
    for pattern in GLOBAL_VAR_PATTERNS:
        if re.match(pattern, symbol_name):
            return True
    return False


def abstract_symbol(symbol_name: str, is_memory_ref: bool = False) -> str:
    """
    Abstract a symbol name based on its type.
    
    Args:
        symbol_name: Original symbol name
        is_memory_ref: True if this is a memory reference (affects abstraction choice)
    
    Returns:
        Abstracted symbol or original if should be preserved
    """
    # Check if should preserve
    if should_preserve_symbol(symbol_name):
        return symbol_name
    
    # Abstract based on type
    if should_abstract_as_global_var(symbol_name):
        return 'GLOBAL_VAR'
    
    if should_abstract_as_local_var(symbol_name):
        return 'LOCAL_VAR'
    
    # For other unrecognized symbols in memory references, abstract as unknown
    if is_memory_ref and not symbol_name.startswith('0x'):
        return 'UNKNOWN_SYM'
    
    # Default: keep as-is (might be register, constant, etc.)
    return symbol_name


def test_abstraction():
    """Test the abstraction logic."""
    test_cases = [
        # (symbol, expected, description)
        ('.malloc_ptr', '.malloc_ptr', 'Library function - preserve'),
        ('.printf_ptr', '.printf_ptr', 'Library function - preserve'),
        ('st_size', 'st_size', 'Standard struct field - preserve'),
        ('tv_sec', 'tv_sec', 'Standard struct field - preserve'),
        ('euid', 'euid', 'Standard struct field - preserve'),
        ('code', 'LOCAL_VAR', 'Local variable - abstract'),
        ('speed', 'LOCAL_VAR', 'Local variable - abstract'),
        ('how', 'LOCAL_VAR', 'Local variable - abstract'),
        ('time0', 'LOCAL_VAR', 'Local variable - abstract'),
        ('tmp', 'LOCAL_VAR', 'Temp variable - abstract'),
        ('i', 'LOCAL_VAR', 'Loop counter - abstract'),
        ('dword_12345678', 'GLOBAL_VAR', 'IDA global - abstract'),
        ('byte_ABCD', 'GLOBAL_VAR', 'IDA global - abstract'),
        ('rax', 'rax', 'Register - keep as-is'),
        ('0x12345', '0x12345', 'Hex address - keep as-is'),
    ]
    
    print("Testing symbol abstraction:")
    print("=" * 80)
    for symbol, expected, desc in test_cases:
        result = abstract_symbol(symbol)
        status = "✓" if result == expected else "✗"
        print(f"{status} {desc}")
        print(f"   {symbol} -> {result} (expected: {expected})")
        if result != expected:
            print(f"   MISMATCH!")
    print("=" * 80)


if __name__ == '__main__':
    test_abstraction()
