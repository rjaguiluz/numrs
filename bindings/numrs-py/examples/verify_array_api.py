
import sys
import os

# Ensure we can import numrs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numrs import Array

def verify_array_api():
    print("Verifying Array API...")

    # 1. Zeros
    print("Testing zeros([2, 3])...", end="")
    a = Array.zeros([2, 3])
    assert a.shape == [2, 3]
    assert all(x == 0.0 for x in a.data)
    print(" OK")

    # 2. Ones
    print("Testing ones([2, 3])...", end="")
    b = Array.ones([2, 3])
    assert b.shape == [2, 3]
    assert all(x == 1.0 for x in b.data)
    print(" OK")

    # 3. Reshape
    print("Testing reshape([3, 2])...", end="")
    c = b.reshape([3, 2])
    assert c.shape == [3, 2]
    assert len(c.data) == 6
    assert all(x == 1.0 for x in c.data)
    print(" OK")
    
    # 4. Data access
    print("Testing data access...", end="")
    data = c.data
    assert len(data) == 6
    assert isinstance(data[0], float)
    print(" OK")

    # 5. Math Ops
    print("Testing math ops...", end="")
    x = Array.ones([2, 2])
    y = Array.ones([2, 2])
    
    # Add
    z = x + y
    assert all(val == 2.0 for val in z.data)
    
    # Sub
    z = x - y
    assert all(val == 0.0 for val in z.data)
    
    # Mul
    z = x * y
    assert all(val == 1.0 for val in z.data)
    
    # Div
    z = x / y
    assert all(val == 1.0 for val in z.data)
    
    z = x @ y
    assert all(val == 2.0 for val in z.data)
    print(" OK")

    # 6. Type Casting (from_vec)
    print("Testing casting from int/double...", end="")
    
    # Int input -> f32 storage
    arr_int = Array(data=[1, 2, 3], shape=[3])
    # check it acts as float internally
    assert arr_int.data[0] == 1.0
    assert isinstance(arr_int.data[0], float)
    
    # Double input -> f32 storage
    # In Python floats are doubles. Standard constructor should now use from_f64
    # We can verify by large precision loss or just general functionality
    arr_dbl = Array(data=[1.5, 2.5], shape=[2])
    assert arr_dbl.data[0] == 1.5

    print(" OK")

    print("\nArray API Verification Successful!")

if __name__ == "__main__":
    verify_array_api()
