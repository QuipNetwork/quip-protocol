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
  console.log('=== CHAIN-SDK SDL TEST ===\n')

  const { SDL } = require('@akashnetwork/chain-sdk')
  const sdl = SDL.fromString(TEST_SDL, 'beta3')
  
  const json = sdl.manifestSortedJSON()
  const hash = crypto.createHash('sha256').update(json).digest('hex')
  
  console.log('manifestSortedJSON length:', json.length)
  console.log('Hash:', hash)
  console.log('Has nulls:', json.includes(':null'))
  console.log('First 300:', json.substring(0, 300))
  console.log('')
  
  // Compare with akashjs
  const { SDL: SDL2 } = require('@akashnetwork/akashjs/build/sdl')
  const sdl2 = SDL2.fromString(TEST_SDL, 'beta3')
  const json2 = sdl2.manifestSortedJSON()
  const hash2 = crypto.createHash('sha256').update(json2).digest('hex')
  
  console.log('akashjs hash:', hash2)
  console.log('chain-sdk hash:', hash)
  console.log('Match:', hash === hash2)
}

main().catch(console.error)
