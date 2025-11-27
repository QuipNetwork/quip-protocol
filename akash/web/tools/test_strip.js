// Test stripEmptyValues function
function stripEmptyValues(obj, key) {
  if (obj === null || obj === undefined) return undefined
  if (typeof obj === 'string' && obj === '') return undefined
  if (key === 'val' && typeof obj === 'number') return String(obj)
  if (Array.isArray(obj)) {
    const filtered = obj.map(item => stripEmptyValues(item)).filter(v => v !== undefined)
    return filtered.length > 0 ? filtered : undefined
  }
  if (typeof obj === 'object') {
    const result = {}
    for (const [k, value] of Object.entries(obj)) {
      const stripped = stripEmptyValues(value, k)
      if (stripped !== undefined) result[k] = stripped
    }
    return Object.keys(result).length > 0 ? result : undefined
  }
  return obj
}

const input = [{"name":"dcloud","services":[{"args":null,"command":null,"count":1,"credentials":null,"env":["TEST"],"name":"test","params":null}]}]

console.log("Input:", JSON.stringify(input))
console.log("Has nulls:", JSON.stringify(input).includes(':null'))

const stripped = stripEmptyValues(input)
console.log("\nStripped:", JSON.stringify(stripped))
console.log("Has nulls:", JSON.stringify(stripped).includes(':null'))
