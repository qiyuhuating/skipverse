import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { crc32 } from '../src/node/crc32.js';

describe('crc32', () => {
  it('matches the IEEE 802.3 reference vector', () => {
    const bytes = new TextEncoder().encode('123456789');
    assert.equal(crc32(bytes), 0xcbf43926);
  });

  it('is stable and order-sensitive', () => {
    const a = new TextEncoder().encode('hello world');
    const b = new TextEncoder().encode('hello worlds');
    assert.equal(crc32(a), crc32(a.slice()));
    assert.notEqual(crc32(a), crc32(b));
  });

  it('handles empty input', () => {
    assert.equal(crc32(new Uint8Array(0)), 0);
  });
});
