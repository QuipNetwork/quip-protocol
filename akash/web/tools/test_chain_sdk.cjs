/**
 * Compare akashjs SDL vs chain-sdk SDL
 */
const crypto = require('crypto')

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
  console.log('=== AKASHJS vs CHAIN-SDK COMPARISON ===\n')

  // 1. Test akashjs
  console.log('1. @akashnetwork/akashjs:')
  try {
    const { SDL } = require('@akashnetwork/akashjs/build/sdl')
    const sdl1 = SDL.fromString(TEST_SDL, 'beta3')
    const json1 = sdl1.manifestSortedJSON()
    const hash1 = crypto.createHash('sha256').update(json1).digest('hex')
    console.log('   manifestSortedJSON length:', json1.length)
    console.log('   Hash:', hash1)
    console.log('   First 200:', json1.substring(0, 200))
  } catch (e) {
    console.log('   Error:', e.message)
  }
  console.log('')

  // 2. Test chain-sdk
  console.log('2. @akashnetwork/chain-sdk:')
  try {
    const { SDL } = require('@akashnetwork/chain-sdk/sdl')
    const sdl2 = SDL.fromString(TEST_SDL, 'beta3')
    const json2 = sdl2.manifestSortedJSON()
    const hash2 = crypto.createHash('sha256').update(json2).digest('hex')
    console.log('   manifestSortedJSON length:', json2.length)
    console.log('   Hash:', hash2)
    console.log('   First 200:', json2.substring(0, 200))
  } catch (e) {
    console.log('   Error:', e.message)
  }
}

main().catch(console.error)
