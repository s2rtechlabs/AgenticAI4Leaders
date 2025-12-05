"""
Test client for Calculator MCP Server (FastMCP)
================================================

This script tests the Calculator MCP server by calling its tools directly.
With FastMCP, you can also test using the MCP Inspector or Claude Desktop.

Usage:
    1. Make sure the server is configured in your MCP client (e.g., Claude Desktop)
    2. Or use the FastMCP CLI: fastmcp dev server.py
    3. Or run this test script for direct function testing
"""

import sys
import math

# Import the tools directly from server for testing
# This is a simple way to test without running a full MCP connection
sys.path.insert(0, '.')
from server import add, subtract, multiply, divide, power, square_root, factorial, evaluate


def run_tests():
    """Run test suite for calculator tools."""
    
    print("=" * 60)
    print("Calculator MCP Server - Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Addition
    print("\n[TEST 1] Addition (5 + 3)...")
    result = add(5, 3)
    if "8" in result:
        print(f"✅ {result}")
        tests_passed += 1
    else:
        print(f"❌ Unexpected: {result}")
        tests_failed += 1
    
    # Test 2: Subtraction
    print("\n[TEST 2] Subtraction (10 - 4)...")
    result = subtract(10, 4)
    if "6" in result:
        print(f"✅ {result}")
        tests_passed += 1
    else:
        print(f"❌ Unexpected: {result}")
        tests_failed += 1
    
    # Test 3: Multiplication
    print("\n[TEST 3] Multiplication (7 × 6)...")
    result = multiply(7, 6)
    if "42" in result:
        print(f"✅ {result}")
        tests_passed += 1
    else:
        print(f"❌ Unexpected: {result}")
        tests_failed += 1
    
    # Test 4: Division
    print("\n[TEST 4] Division (20 ÷ 4)...")
    result = divide(20, 4)
    if "5" in result:
        print(f"✅ {result}")
        tests_passed += 1
    else:
        print(f"❌ Unexpected: {result}")
        tests_failed += 1
    
    # Test 5: Division by zero (error handling)
    print("\n[TEST 5] Division by zero (error handling)...")
    result = divide(10, 0)
    if "Error" in result:
        print(f"✅ Error handled: {result}")
        tests_passed += 1
    else:
        print(f"❌ Should have returned error")
        tests_failed += 1
    
    # Test 6: Power
    print("\n[TEST 6] Power (2^8)...")
    result = power(2, 8)
    if "256" in result:
        print(f"✅ {result}")
        tests_passed += 1
    else:
        print(f"❌ Unexpected: {result}")
        tests_failed += 1
    
    # Test 7: Square root
    print("\n[TEST 7] Square root (√144)...")
    result = square_root(144)
    if "12" in result:
        print(f"✅ {result}")
        tests_passed += 1
    else:
        print(f"❌ Unexpected: {result}")
        tests_failed += 1
    
    # Test 8: Square root of negative (error handling)
    print("\n[TEST 8] Square root of negative (error handling)...")
    result = square_root(-16)
    if "Error" in result:
        print(f"✅ Error handled: {result}")
        tests_passed += 1
    else:
        print(f"❌ Should have returned error")
        tests_failed += 1
    
    # Test 9: Factorial
    print("\n[TEST 9] Factorial (5!)...")
    result = factorial(5)
    if "120" in result:
        print(f"✅ {result}")
        tests_passed += 1
    else:
        print(f"❌ Unexpected: {result}")
        tests_failed += 1
    
    # Test 10: Expression evaluation
    print("\n[TEST 10] Expression evaluation (sqrt(16) + pow(2, 3))...")
    result = evaluate("sqrt(16) + pow(2, 3)")
    if "12" in result:
        print(f"✅ {result}")
        tests_passed += 1
    else:
        print(f"❌ Unexpected: {result}")
        tests_failed += 1
    
    # Test 11: Trigonometry
    print("\n[TEST 11] Trigonometry (sin(pi/2))...")
    result = evaluate("sin(pi/2)")
    if "1" in result:
        print(f"✅ {result}")
        tests_passed += 1
    else:
        print(f"❌ Unexpected: {result}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    if tests_failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {tests_failed} test(s) failed")
    
    return tests_failed == 0


if __name__ == "__main__":
    print("\n📝 Testing Calculator MCP Server (FastMCP)")
    print("\nNote: This tests the tool functions directly.")
    print("For full MCP testing, use:")
    print("  - fastmcp dev server.py (MCP Inspector)")
    print("  - Claude Desktop with MCP configured")
    print()
    
    success = run_tests()
    sys.exit(0 if success else 1)
# AI Generated Code by Deloitte + Cursor (END)
