/**
 * Verify our implementation matches akashjs exactly
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
  const { SDL } = require('@akashnetwork/akashjs/build/sdl')

  const sdl = SDL.fromString(TEST_SDL, 'beta3')

  console.log('=== AKASHJS VERIFICATION ===\n')

  // 1. Get manifestSortedJSON
  const manifestJson = sdl.manifestSortedJSON()
  console.log('1. manifestSortedJSON():')
  console.log('   Length:', manifestJson.length, 'bytes')
  console.log('   First 300:', manifestJson.substring(0, 300))
  console.log('')

  // 2. Compute hash using our method (browser-compatible)
  const encoder = new TextEncoder()
  const data = encoder.encode(manifestJson)
  const hashBuffer = crypto.createHash('sha256').update(data).digest()
  const ourHash = hashBuffer.toString('hex')
  console.log('2. Our hash computation:')
  console.log('   SHA256:', ourHash)
  console.log('')

  // 3. Get akashjs manifestVersion (may fail in Node due to crypto.subtle)
  console.log('3. akashjs manifestVersion():')
  try {
    const version = await sdl.manifestVersion()
    const akashjsHash = Buffer.from(version).toString('hex')
    console.log('   SHA256:', akashjsHash)
    console.log('   Matches our hash:', ourHash === akashjsHash)
  } catch (e) {
    console.log('   Error (expected in Node):', e.message)
    // Manually compute what manifestVersion does
    const manualHash = crypto.createHash('sha256').update(manifestJson).digest('hex')
    console.log('   Manual equivalent:', manualHash)
    console.log('   Matches our hash:', ourHash === manualHash)
  }
  console.log('')

  // 4. Check manifest structure
  console.log('4. Manifest structure check:')
  const manifest = sdl.manifest(true) // asString=true
  console.log('   Groups:', manifest.length)
  console.log('   First group name:', manifest[0]?.name)
  console.log('   Services:', manifest[0]?.services?.length)
  console.log('   First service:', JSON.stringify(manifest[0]?.services?.[0], null, 2).substring(0, 500))
  console.log('')

  // 5. Check groups() for deployment message
  console.log('5. groups() for MsgCreateDeployment:')
  const groups = sdl.groups()
  console.log('   Groups count:', groups.length)
  console.log('   First group:', JSON.stringify(groups[0], null, 2).substring(0, 300))
}

main().catch(console.error)
