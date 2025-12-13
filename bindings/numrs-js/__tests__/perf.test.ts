import * as numrs from '../index';

function timeIt(label: string, fn: () => void, iterations: number = 200): number {
  const start = Date.now();
  for (let i = 0; i < iterations; i++) fn();
  const total = Date.now() - start;
  const perOp = total / iterations;
  console.log(`${label}: ${total}ms total, avg ${perOp.toFixed(3)}ms/op (${iterations} iter)`);
  return perOp;
}

describe('Performance Tests (Informational) — TypeScript API', () => {
  test('perf: binary ops and matmul with Float32Array', () => {
    // Test with reasonably sized arrays
    const size = 512;
    const matSize = size * size;
    
    // Generate random data
    const aData = new Float32Array(matSize);
    const bData = new Float32Array(matSize);
    for (let i = 0; i < matSize; i++) {
      aData[i] = Math.random();
      bData[i] = Math.random();
    }

    // Benchmark add
    const pAdd = timeIt('add 512x512', () => {
      numrs.add(aData, bData);
    }, 100);

    // Benchmark mul
    const pMul = timeIt('mul 512x512', () => {
      numrs.mul(aData, bData);
    }, 100);

    // Benchmark matmul (fewer iterations, it's expensive)
    const pMat = timeIt('matmul 512x512', () => {
      numrs.matmul(aData, bData, { shape: [size, size] });
    }, 10);

    // Just verify they returned numbers
    expect(typeof pAdd).toBe('number');
    expect(typeof pMul).toBe('number');
    expect(typeof pMat).toBe('number');
    
    // Sanity checks (loose bounds)
    expect(pAdd).toBeLessThan(1000); // Should be under 1 second
    expect(pMul).toBeLessThan(1000);
    expect(pMat).toBeLessThan(5000); // matmul can be slower
  }, 120000); // 2 minute timeout

  test('perf: reductions on large arrays', () => {
    const size = 1000000; // 1M elements
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.random();
    }

    const pSum = timeIt('sum 1M', () => {
      numrs.sum(data);
    }, 50);

    const pMean = timeIt('mean 1M', () => {
      numrs.mean(data);
    }, 50);

    expect(typeof pSum).toBe('number');
    expect(typeof pMean).toBe('number');
    expect(pSum).toBeLessThan(100); // Should be very fast
    expect(pMean).toBeLessThan(100);
  }, 60000);

  test('perf: unary operations', () => {
    const size = 100000;
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.random();
    }

    const pSin = timeIt('sin 100K', () => {
      numrs.sin(data);
    }, 100);

    const pExp = timeIt('exp 100K', () => {
      numrs.exp(data);
    }, 100);

    expect(typeof pSin).toBe('number');
    expect(typeof pExp).toBe('number');
  }, 60000);
});
