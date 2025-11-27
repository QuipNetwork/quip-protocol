/**
 * Full manifest hash comparison test
 *
 * This compares:
 * 1. akashjs manifestSortedJSON() output
 * 2. After stripping nulls (to match Go's omitempty)
 * 3. What Go produces after unmarshal/remarshal
 */

const { execSync } = require('child_process')
const crypto = require('crypto')

// SDL for testing
const TEST_SDL = `version: "2.0"
services:
  web:
    image: nginx:alpine
    expose:
      - port: 80
        as: 80
        to:
          - global: true
profiles:
  compute:
    web:
      resources:
        cpu:
          units: 0.5
        memory:
          size: 512Mi
        storage:
          size: 1Gi
  placement:
    dcloud:
      pricing:
        web:
          denom: uakt
          amount: 1000
deployment:
  web:
    dcloud:
      profile: web
      count: 1
`

async function main() {
  console.log('=== FULL MANIFEST HASH COMPARISON TEST ===\n')

  // Load akashjs
  const { SDL } = require('@akashnetwork/akashjs/build/sdl')

  // Parse SDL
  const sdl = SDL.fromString(TEST_SDL, 'beta3')

  // Method 1: akashjs manifestSortedJSON()
  const manifestJson = sdl.manifestSortedJSON()
  console.log('1. akashjs manifestSortedJSON():')
  console.log('   Length:', manifestJson.length, 'bytes')
  console.log('   Has nulls:', manifestJson.includes(':null'))
  console.log('   First 200:', manifestJson.substring(0, 200))

  const hash1 = crypto.createHash('sha256').update(manifestJson).digest('hex')
  console.log('   SHA256:', hash1)

  // Method 2: Strip nulls and empty values
  const parsed = JSON.parse(manifestJson)
  const stripped = stripEmptyValues(parsed)

  // Re-serialize with stable key ordering
  const strippedJson = stableStringify(stripped)

  console.log('\n2. After stripping nulls:')
  console.log('   Length:', strippedJson.length, 'bytes')
  console.log('   Has nulls:', strippedJson.includes(':null'))
  console.log('   First 200:', strippedJson.substring(0, 200))

  const hash2 = crypto.createHash('sha256').update(strippedJson).digest('hex')
  console.log('   SHA256:', hash2)

  // Method 3: Run Go tool to see what provider produces
  console.log('\n3. Go unmarshal/remarshal simulation:')

  // Write stripped JSON to temp file
  const fs = require('fs')
  const tempFile = '/tmp/manifest_test.json'
  fs.writeFileSync(tempFile, strippedJson)

  try {
    // Run Go tool
    const goOutput = execSync(
      `cd /Users/carback1/Code/quip-protocol/akash/web/tools && go run akash_manifest_hash.go < ${tempFile}`,
      { encoding: 'utf8' }
    )
    console.log(goOutput)
  } catch (error) {
    console.log('   Go tool error:', error.message)
    // Still show stdout if available
    if (error.stdout) console.log(error.stdout)
    if (error.stderr) console.log('   stderr:', error.stderr)
  }

  // Method 4: Test HTML escaping
  console.log('\n4. HTML escaping check:')
  console.log('   Has \\u003c (escaped <):', manifestJson.includes('\\u003c'))
  console.log('   Has \\u003e (escaped >):', manifestJson.includes('\\u003e'))
  console.log('   Has \\u0026 (escaped &):', manifestJson.includes('\\u0026'))

  // Method 5: Compare exact bytes
  console.log('\n5. Byte comparison of first difference:')
  if (manifestJson !== strippedJson) {
    findFirstDiff(manifestJson, strippedJson)
  } else {
    console.log('   Identical!')
  }

  console.log('\n=== SUMMARY ===')
  console.log('Hash with nulls:   ', hash1)
  console.log('Hash without nulls:', hash2)
  console.log('Match:', hash1 === hash2)
}

/**
 * Strip null and empty values (matches Go's omitempty behavior)
 */
function stripEmptyValues(obj, key) {
  if (obj === null || obj === undefined) return undefined
  if (typeof obj === 'string' && obj === '') return undefined

  // Convert 'val' fields to strings (Go's ResourceUnits.val is string type)
  if (key === 'val' && typeof obj === 'number') {
    return String(obj)
  }

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

/**
 * Stable JSON stringify with sorted keys and HTML escaping
 */
function stableStringify(obj) {
  return JSON.stringify(sortKeys(obj))
    .replace(/</g, '\\u003c')
    .replace(/>/g, '\\u003e')
    .replace(/&/g, '\\u0026')
}

function sortKeys(obj) {
  if (obj === null || obj === undefined) return obj
  if (Array.isArray(obj)) return obj.map(sortKeys)
  if (typeof obj !== 'object') return obj

  const sorted = {}
  Object.keys(obj).sort().forEach(key => {
    sorted[key] = sortKeys(obj[key])
  })
  return sorted
}

function findFirstDiff(a, b) {
  const minLen = Math.min(a.length, b.length)
  for (let i = 0; i < minLen; i++) {
    if (a[i] !== b[i]) {
      const start = Math.max(0, i - 30)
      const end = Math.min(minLen, i + 30)
      console.log(`   Position ${i}:`)
      console.log(`   Original:  ...${a.substring(start, end)}...`)
      console.log(`   Stripped:  ...${b.substring(start, end)}...`)
      return
    }
  }
  if (a.length !== b.length) {
    console.log(`   Length diff: ${a.length} vs ${b.length}`)
  }
}

main().catch(console.error)
