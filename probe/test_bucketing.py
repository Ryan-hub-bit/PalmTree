"""
Test BB bucket assignment logic

Verify that bb_norm values are correctly mapped to buckets.
"""

def get_bb_bucket(bb_norm, num_buckets=10):
    """Assign instruction to a bucket based on its position within BB."""
    bucket = int(bb_norm * num_buckets)
    if bucket >= num_buckets:
        bucket = num_buckets - 1
    return bucket


# Test cases
test_cases = [
    (0.00, 0),  # Beginning of BB
    (0.05, 0),  # Still in bucket 0
    (0.10, 1),  # Start of bucket 1
    (0.12, 1),  # Your example
    (0.20, 2),  # Bucket 2
    (0.25, 2),  # Mid bucket 2
    (0.50, 5),  # Middle of BB
    (0.75, 7),  # 3/4 through BB
    (0.90, 9),  # Near end
    (0.95, 9),  # End of BB
    (1.00, 9),  # Exactly at end (edge case)
]

print("BB Bucket Assignment Tests")
print("=" * 50)
print(f"{'bb_norm':>10} | {'Expected':>10} | {'Actual':>10} | {'Status':>10}")
print("-" * 50)

all_passed = True
for bb_norm, expected_bucket in test_cases:
    actual_bucket = get_bb_bucket(bb_norm, num_buckets=10)
    status = "✓ PASS" if actual_bucket == expected_bucket else "✗ FAIL"
    if actual_bucket != expected_bucket:
        all_passed = False
    print(f"{bb_norm:>10.2f} | {expected_bucket:>10} | {actual_bucket:>10} | {status:>10}")

print("=" * 50)
if all_passed:
    print("✓ All tests passed!")
else:
    print("✗ Some tests failed!")
