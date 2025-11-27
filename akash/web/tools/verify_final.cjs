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
  console.log('=== FINAL VERIFICATION (using chain-sdk) ===\n')

  const { SDL } = require('@akashnetwork/chain-sdk')
  const sdl = SDL.fromString(TEST_SDL, 'beta3')

  const json = sdl.manifestSortedJSON()
  const hash = crypto.createHash('sha256').update(json).digest('hex')

  console.log('1. chain-sdk manifestSortedJSON():')
  console.log('   Length:', json.length, 'bytes')
  console.log('   Hash:', hash)
  console.log('   Has nulls:', json.includes(':null'))
  console.log('   Has params:null:', json.includes('"params":null'))
  console.log('')

  // Write to file and test with Go
  const fs = require('fs')
  fs.writeFileSync('/tmp/manifest_chainsdk.json', json)

  console.log('2. Go validation:')
  const { execSync } = require('child_process')
  try {
    const goOutput = execSync(
      `cd /Users/carback1/Code/quip-protocol/akash/web/tools && go run akash_manifest_hash.go < /tmp/manifest_chainsdk.json`,
      { encoding: 'utf8' }
    )
    console.log(goOutput)
  } catch (error) {
    console.log('Go error:', error.message)
    if (error.stdout) console.log(error.stdout)
  }
}

main().catch(console.error)
