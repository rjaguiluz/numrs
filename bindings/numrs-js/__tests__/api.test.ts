import * as numrs from '../index';

describe('NumRs TypeScript API — Smoke Tests', () => {
  test('binary operations (add, mul) with Float32Array', () => {
    // Create input data
    const a = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const b = new Float32Array([2.0, 2.0, 2.0, 2.0]);
    
    // Add operation
    const added = numrs.add(a, b);
    expect(added).toBeInstanceOf(Float32Array);
    expect(Array.from(added)).toEqual([3.0, 4.0, 5.0, 6.0]);
    
    // Multiply operation
    const multiplied = numrs.mul(a, b);
    expect(multiplied).toBeInstanceOf(Float32Array);
    expect(Array.from(multiplied)).toEqual([2.0, 4.0, 6.0, 8.0]);
  });

  test('unary operations (sin, abs, neg)', () => {
    const data = new Float32Array([1.0, -2.0, 3.0, -4.0]);
    
    // Absolute value
    const absResult = numrs.abs(data);
    expect(absResult).toBeInstanceOf(Float32Array);
    expect(Array.from(absResult)).toEqual([1.0, 2.0, 3.0, 4.0]);
    
    // Negation
    const negResult = numrs.neg(data);
    expect(negResult).toBeInstanceOf(Float32Array);
    expect(Array.from(negResult)).toEqual([-1.0, 2.0, -3.0, 4.0]);
    
    // Sin (just check it runs)
    const sinResult = numrs.sin(data);
    expect(sinResult).toBeInstanceOf(Float32Array);
    expect(sinResult.length).toBe(4);
  });

  test('reduction operations (sum, mean, max)', () => {
    const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    
    // Sum
    const sumResult = numrs.sum(data);
    expect(sumResult).toBeInstanceOf(Float32Array);
    expect(sumResult[0]).toBeCloseTo(10.0, 5);
    
    // Mean
    const meanResult = numrs.mean(data);
    expect(meanResult).toBeInstanceOf(Float32Array);
    expect(meanResult[0]).toBeCloseTo(2.5, 5);
    
    // Max
    const maxResult = numrs.max(data);
    expect(maxResult).toBeInstanceOf(Float32Array);
    expect(maxResult[0]).toBe(4.0);
    
    // Min
    const minResult = numrs.min(data);
    expect(minResult).toBeInstanceOf(Float32Array);
    expect(minResult[0]).toBe(1.0);
  });

  test('matmul works for simple 2x2 matrices', () => {
    // Matrix A = [[1, 2], [3, 4]] (row-major)
    const a = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    // Matrix B = [[1, 0], [0, 1]] (identity, row-major)
    const b = new Float32Array([1.0, 0.0, 0.0, 1.0]);
    
    const result = numrs.matmul(a, b, { shape: [2, 2] });
    expect(result).toBeInstanceOf(Float32Array);
    // Result should be A (multiplying by identity)
    expect(Array.from(result)).toEqual([1.0, 2.0, 3.0, 4.0]);
  });

  test('dot product', () => {
    const a = new Float32Array([1.0, 2.0, 3.0]);
    const b = new Float32Array([4.0, 5.0, 6.0]);
    
    const result = numrs.dot(a, b);
    expect(result).toBeInstanceOf(Float32Array);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    expect(result[0]).toBeCloseTo(32.0, 5);
  });

  test('transpose with shape option', () => {
    // Matrix [[1, 2], [3, 4]]
    const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    
    const result = numrs.transpose(data, { shape: [2, 2] });
    expect(result).toBeInstanceOf(Float32Array);
    // Transposed: [[1, 3], [2, 4]]
    expect(Array.from(result)).toEqual([1.0, 3.0, 2.0, 4.0]);
  });

  test('backend info returns valid structure', () => {
    const info = numrs.backendInfo();
    expect(info).toHaveProperty('selected');
    expect(info).toHaveProperty('add');
    expect(info).toHaveProperty('matmul');
    expect(typeof info.selected).toBe('string');
  });

  test('startup log returns boolean', () => {
    const result = numrs.startupLog();
    expect(typeof result).toBe('boolean');
  });
});
