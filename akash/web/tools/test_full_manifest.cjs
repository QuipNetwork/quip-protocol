// Test full manifest transformation pipeline
const fs = require('fs')

// Our stripEmptyValues function
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

// Simulated akashjs manifestSortedJSON output (with nulls, sorted keys)
const akashjsOutput = [
  {
    "name": "dcloud",
    "services": [
      {
        "args": null,
        "command": null,
        "count": 1,
        "credentials": null,
        "env": [
          "MINER_TYPE=cpu",
          "MINING_DURATION=5m",
          "DIFFICULTY_ENERGY=-14500",
          "MIN_DIVERSITY=0.15",
          "MIN_SOLUTIONS=5",
          "REQUESTED_CPUS=1",
          "REQUESTED_GPUS=0"
        ],
        "expose": [
          {
            "endpointSequenceNumber": 0,
            "externalPort": 80,
            "global": true,
            "hosts": null,
            "httpOptions": {
              "maxBodySize": 1048576,
              "nextCases": ["error", "timeout"],
              "nextTimeout": 0,
              "nextTries": 3,
              "readTimeout": 60000,
              "sendTimeout": 60000
            },
            "ip": "",
            "port": 8080,
            "proto": "TCP",
            "service": ""
          }
        ],
        "image": "carback1/quip-protocol-cpu-miner:latest",
        "name": "quip-cpu-miner",
        "params": null,
        "resources": {
          "cpu": {
            "units": {
              "val": "1000"
            }
          },
          "endpoints": [
            {
              "sequence_number": 0
            }
          ],
          "gpu": {
            "units": {
              "val": "0"
            }
          },
          "id": 1,
          "memory": {
            "size": {
              "val": "2147483648"
            }
          },
          "storage": [
            {
              "name": "default",
              "size": {
                "val": "5368709120"
              }
            }
          ]
        }
      }
    ]
  }
]

console.log("=== Manifest Transformation Test ===\n")

// Step 1: What akashjs produces (with nulls)
const akashjsJson = JSON.stringify(akashjsOutput)
console.log("1. akashjs output (with nulls):")
console.log("   Length:", akashjsJson.length)
console.log("   Has nulls:", akashjsJson.includes(':null'))
console.log("   First 100:", akashjsJson.substring(0, 100) + "...")

// Step 2: After stripping nulls
const stripped = stripEmptyValues(akashjsOutput)
const strippedJson = JSON.stringify(stripped)
console.log("\n2. After stripEmptyValues:")
console.log("   Length:", strippedJson.length)
console.log("   Has nulls:", strippedJson.includes(':null'))
console.log("   First 100:", strippedJson.substring(0, 100) + "...")

// Step 3: With HTML escaping (matching Go's json encoder)
const escapedJson = strippedJson
  .replace(/</g, '\\u003c')
  .replace(/>/g, '\\u003e')
  .replace(/&/g, '\\u0026')
console.log("\n3. After HTML escaping:")
console.log("   Length:", escapedJson.length)
console.log("   First 100:", escapedJson.substring(0, 100) + "...")

// Step 4: Compute hash
const crypto = require('crypto')
const hash = crypto.createHash('sha256').update(escapedJson).digest('hex')
console.log("\n4. SHA256 hash:", hash)

// Write to file for Go comparison
fs.writeFileSync('test_manifest_stripped.json', escapedJson)
console.log("\n5. Written to test_manifest_stripped.json")
console.log("   Run: cat test_manifest_stripped.json | go run akash_manifest_hash.go")
