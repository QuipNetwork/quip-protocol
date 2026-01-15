/**
 * Certificate generation utilities for Akash Network mTLS authentication
 *
 * Generates x509 certificates compatible with Akash Network's certificate requirements.
 * Uses ECDSA P-256 with SHA-256 for signing.
 *
 * References:
 * - https://akash.network/blog/how-to-create-an-x509-certificate-in-browser-for-fun-and-profit/
 * - https://docs.akash.network/other-resources/mtls
 */

import * as asn1js from 'asn1js'
import * as pkijs from 'pkijs'
import { arrayBufferToString, toBase64 } from 'pvutils'

// Storage key prefix for certificates
const CERT_STORAGE_PREFIX = 'akash_cert_'

export interface AkashCertificate {
  certPem: string
  privateKeyPem: string
  pubkeyPem: string
  serialNumber: string
  validFrom: Date
  validTo: Date
}

/**
 * Format a base64 string as PEM with proper line breaks and headers
 */
function formatPEM(pemString: string, type: string): string {
  const header = `-----BEGIN ${type}-----`
  const footer = `-----END ${type}-----`
  // Split into 64-character lines
  const lines = pemString.match(/.{1,64}/g) || []
  return `${header}\n${lines.join('\n')}\n${footer}`
}

/**
 * Get the Web Crypto API instance
 */
function getWebCrypto(): SubtleCrypto {
  return crypto.subtle
}

/**
 * Generate an ECDSA key pair for certificate signing
 */
async function generateKeyPair(): Promise<CryptoKeyPair> {
  const crypto = getWebCrypto()

  return crypto.generateKey(
    {
      name: 'ECDSA',
      namedCurve: 'P-256'
    },
    true, // extractable
    ['sign', 'verify']
  )
}

/**
 * Export a public key to SPKI format (for on-chain storage)
 */
async function exportPublicKeyPEM(publicKey: CryptoKey): Promise<string> {
  const crypto = getWebCrypto()
  const exported = await crypto.exportKey('spki', publicKey)
  const base64 = toBase64(arrayBufferToString(exported))
  return formatPEM(base64, 'PUBLIC KEY')
}

/**
 * Export a private key to PKCS8 format (for local storage only)
 */
async function exportPrivateKeyPEM(privateKey: CryptoKey): Promise<string> {
  const crypto = getWebCrypto()
  const exported = await crypto.exportKey('pkcs8', privateKey)
  const base64 = toBase64(arrayBufferToString(exported))
  return formatPEM(base64, 'PRIVATE KEY')
}

/**
 * Generate an x509 certificate for Akash Network
 *
 * @param address - The Akash wallet address (used as CN)
 * @param validityDays - Certificate validity in days (default: 365)
 */
export async function generateCertificate(
  address: string,
  validityDays: number = 365
): Promise<AkashCertificate> {
  // Set up pkijs crypto engine
  const cryptoEngine = new pkijs.CryptoEngine({
    name: 'WebCrypto',
    crypto: crypto,
    subtle: crypto.subtle
  })
  pkijs.setEngine('WebCrypto', cryptoEngine)

  // Generate key pair
  const keyPair = await generateKeyPair()

  // Create certificate
  const certificate = new pkijs.Certificate()

  // Set version to v3 (value 2 = version 3)
  certificate.version = 2

  // Set serial number to current timestamp
  const serialNumber = Date.now()
  certificate.serialNumber = new asn1js.Integer({ value: serialNumber })

  // Set validity period
  const validFrom = new Date()
  const validTo = new Date()
  validTo.setDate(validTo.getDate() + validityDays)

  certificate.notBefore.value = validFrom
  certificate.notAfter.value = validTo

  // Set issuer (self-signed, so issuer = subject)
  // Using the Akash address as the Common Name
  certificate.issuer.typesAndValues.push(new pkijs.AttributeTypeAndValue({
    type: '2.5.4.3', // OID for Common Name
    value: new asn1js.Utf8String({ value: address })
  }))

  // Set subject (same as issuer for self-signed)
  certificate.subject.typesAndValues.push(new pkijs.AttributeTypeAndValue({
    type: '2.5.4.3', // OID for Common Name
    value: new asn1js.Utf8String({ value: address })
  }))

  // Add extensions
  certificate.extensions = []

  // Basic Constraints - not a CA
  certificate.extensions.push(new pkijs.Extension({
    extnID: '2.5.29.19', // OID for Basic Constraints
    critical: true,
    extnValue: new asn1js.Sequence({
      value: [
        new asn1js.Boolean({ value: false }) // cA = false
      ]
    }).toBER(false)
  }))

  // Key Usage - digital signature
  const keyUsageBits = new ArrayBuffer(1)
  const keyUsageView = new Uint8Array(keyUsageBits)
  keyUsageView[0] = 0x80 // digitalSignature

  certificate.extensions.push(new pkijs.Extension({
    extnID: '2.5.29.15', // OID for Key Usage
    critical: true,
    extnValue: new asn1js.BitString({ valueHex: keyUsageBits }).toBER(false)
  }))

  // Extended Key Usage - clientAuth
  certificate.extensions.push(new pkijs.Extension({
    extnID: '2.5.29.37', // OID for Extended Key Usage
    critical: false,
    extnValue: new asn1js.Sequence({
      value: [
        new asn1js.ObjectIdentifier({ value: '1.3.6.1.5.5.7.3.2' }) // clientAuth
      ]
    }).toBER(false)
  }))

  // Import the public key into the certificate
  await certificate.subjectPublicKeyInfo.importKey(keyPair.publicKey)

  // Sign the certificate with the private key
  await certificate.sign(keyPair.privateKey, 'SHA-256')

  // Export certificate to PEM
  const certDer = certificate.toSchema(true).toBER(false)
  const certBase64 = toBase64(arrayBufferToString(certDer))
  const certPem = formatPEM(certBase64, 'CERTIFICATE')

  // Export private key to PEM
  const privateKeyPem = await exportPrivateKeyPEM(keyPair.privateKey)

  // Export public key from certificate's SubjectPublicKeyInfo (ensures exact match)
  // This is more reliable than re-exporting via Web Crypto
  const spkiDer = certificate.subjectPublicKeyInfo.toSchema().toBER(false)
  const spkiBase64 = toBase64(arrayBufferToString(spkiDer))
  // Akash expects "EC PUBLIC KEY" header for ECDSA keys (SEC1/ANSI X9.62 format)
  // Note: Some versions of Akash also accept "PUBLIC KEY" (SPKI format)
  const pubkeyPem = formatPEM(spkiBase64, 'EC PUBLIC KEY')

  console.log('Generated certificate PEM length:', certPem.length)
  console.log('Generated pubkey PEM length:', pubkeyPem.length)
  console.log('Full pubkey PEM:', pubkeyPem)

  return {
    certPem,
    privateKeyPem,
    pubkeyPem,
    serialNumber: serialNumber.toString(),
    validFrom,
    validTo
  }
}

/**
 * Store certificate in localStorage
 */
export function storeCertificate(address: string, cert: AkashCertificate): void {
  const key = `${CERT_STORAGE_PREFIX}${address}`
  localStorage.setItem(key, JSON.stringify({
    ...cert,
    validFrom: cert.validFrom.toISOString(),
    validTo: cert.validTo.toISOString()
  }))
}

/**
 * Load certificate from localStorage
 */
export function loadCertificate(address: string): AkashCertificate | null {
  const key = `${CERT_STORAGE_PREFIX}${address}`
  const stored = localStorage.getItem(key)

  if (!stored) {
    return null
  }

  try {
    const parsed = JSON.parse(stored)
    return {
      ...parsed,
      validFrom: new Date(parsed.validFrom),
      validTo: new Date(parsed.validTo)
    }
  } catch {
    console.error('Failed to parse stored certificate')
    return null
  }
}

/**
 * Remove certificate from localStorage
 */
export function removeCertificate(address: string): void {
  const key = `${CERT_STORAGE_PREFIX}${address}`
  localStorage.removeItem(key)
}

/**
 * Check if a certificate is still valid (not expired)
 */
export function isCertificateValid(cert: AkashCertificate): boolean {
  const now = new Date()
  return now >= cert.validFrom && now <= cert.validTo
}

/**
 * Convert PEM to bytes for on-chain submission
 * The Akash blockchain stores the raw PEM as bytes
 */
export function pemToBytes(pem: string): Uint8Array {
  return new TextEncoder().encode(pem)
}

/**
 * Convert bytes to base64 for message encoding
 */
export function bytesToBase64(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes))
}
