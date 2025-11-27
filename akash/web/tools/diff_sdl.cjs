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

const { SDL } = require('@akashnetwork/chain-sdk')
const { SDL: SDL2 } = require('@akashnetwork/akashjs/build/sdl')

const sdl1 = SDL.fromString(TEST_SDL, 'beta3')
const sdl2 = SDL2.fromString(TEST_SDL, 'beta3')

const json1 = sdl1.manifestSortedJSON()
const json2 = sdl2.manifestSortedJSON()

console.log('chain-sdk length:', json1.length)
console.log('akashjs length:', json2.length)
console.log('')

// Find first difference
for (let i = 0; i < Math.min(json1.length, json2.length); i++) {
  if (json1[i] !== json2[i]) {
    const start = Math.max(0, i - 50)
    const end = Math.min(json1.length, json2.length, i + 50)
    console.log('First difference at position', i)
    console.log('chain-sdk: ...' + json1.substring(start, end) + '...')
    console.log('akashjs:   ...' + json2.substring(start, end) + '...')
    break
  }
}

// Show full JSON for comparison
console.log('\n=== FULL chain-sdk JSON ===')
console.log(json1)
console.log('\n=== FULL akashjs JSON ===')
console.log(json2)
