/**
 * Vite plugin to proxy manifest requests to Akash providers with mTLS
 *
 * This runs on the Node.js side of the Vite dev server, allowing us to use
 * https.Agent with client certificates for mTLS authentication - something
 * browsers can't do directly.
 *
 * Usage in browser:
 *   POST /api/provider-proxy
 *   {
 *     "url": "https://provider.host/deployment/{dseq}/manifest",
 *     "method": "PUT",
 *     "body": "{...manifest JSON...}",
 *     "certPem": "-----BEGIN CERTIFICATE-----...",
 *     "keyPem": "-----BEGIN PRIVATE KEY-----..."
 *   }
 */

import type { Plugin, Connect } from 'vite'
import https from 'https'
import { IncomingMessage, ServerResponse } from 'http'

interface ProxyRequest {
  url: string
  method: 'GET' | 'POST' | 'PUT' | 'DELETE'
  body?: string
  certPem: string
  keyPem: string
  timeout?: number
}

interface ProxyResponse {
  success: boolean
  status?: number
  statusText?: string
  body?: string
  error?: string
  durationMs: number
}

async function handleProxyRequest(
  reqBody: ProxyRequest
): Promise<ProxyResponse> {
  const startTime = Date.now()

  const { url, method, body, certPem, keyPem, timeout = 30000 } = reqBody

  console.log('[provider-proxy] Request:', { url, method, bodyLength: body?.length })

  return new Promise((resolve) => {
    try {
      const parsedUrl = new URL(url)

      const agent = new https.Agent({
        cert: certPem,
        key: keyPem,
        rejectUnauthorized: false, // Providers use self-signed certs
        servername: '', // Disable SNI for mTLS
        timeout
      })

      const options: https.RequestOptions = {
        hostname: parsedUrl.hostname,
        port: parsedUrl.port || 443,
        path: parsedUrl.pathname + parsedUrl.search,
        method,
        agent,
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': body ? Buffer.byteLength(body) : 0
        },
        timeout
      }

      console.log('[provider-proxy] Connecting to:', parsedUrl.hostname)

      const req = https.request(options, (res) => {
        let responseBody = ''

        res.on('data', (chunk) => {
          responseBody += chunk
        })

        res.on('end', () => {
          const durationMs = Date.now() - startTime
          console.log('[provider-proxy] Response:', {
            status: res.statusCode,
            statusMessage: res.statusMessage,
            bodyLength: responseBody.length,
            durationMs
          })

          resolve({
            success: res.statusCode !== undefined && res.statusCode >= 200 && res.statusCode < 300,
            status: res.statusCode,
            statusText: res.statusMessage,
            body: responseBody,
            durationMs
          })

          agent.destroy()
        })

        res.on('error', (error) => {
          const durationMs = Date.now() - startTime
          console.error('[provider-proxy] Response error:', error)
          resolve({
            success: false,
            error: `Response error: ${error.message}`,
            durationMs
          })
          agent.destroy()
        })
      })

      req.on('error', (error) => {
        const durationMs = Date.now() - startTime
        console.error('[provider-proxy] Request error:', error)
        resolve({
          success: false,
          error: `Request error: ${error.message}`,
          durationMs
        })
        agent.destroy()
      })

      req.on('timeout', () => {
        const durationMs = Date.now() - startTime
        console.error('[provider-proxy] Request timeout')
        req.destroy()
        resolve({
          success: false,
          error: `Request timeout after ${timeout}ms`,
          durationMs
        })
        agent.destroy()
      })

      if (body && method !== 'GET') {
        req.write(body)
      }

      req.end()
    } catch (error) {
      const durationMs = Date.now() - startTime
      const message = error instanceof Error ? error.message : String(error)
      console.error('[provider-proxy] Error:', message)
      resolve({
        success: false,
        error: message,
        durationMs
      })
    }
  })
}

function parseRequestBody(req: Connect.IncomingMessage): Promise<ProxyRequest> {
  return new Promise((resolve, reject) => {
    let body = ''
    req.on('data', (chunk) => {
      body += chunk
    })
    req.on('end', () => {
      try {
        resolve(JSON.parse(body))
      } catch (e) {
        reject(new Error('Invalid JSON body'))
      }
    })
    req.on('error', reject)
  })
}

/**
 * Vite plugin that adds a provider proxy middleware
 */
export function providerProxyPlugin(): Plugin {
  return {
    name: 'vite-provider-proxy',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        // Only handle POST requests to /api/provider-proxy
        if (req.url !== '/api/provider-proxy' || req.method !== 'POST') {
          return next()
        }

        // Set CORS headers
        res.setHeader('Access-Control-Allow-Origin', '*')
        res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS')
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type')

        // Handle preflight
        if (req.method === 'OPTIONS') {
          res.statusCode = 204
          res.end()
          return
        }

        try {
          const proxyReq = await parseRequestBody(req)

          // Validate required fields
          if (!proxyReq.url || !proxyReq.method || !proxyReq.certPem || !proxyReq.keyPem) {
            res.statusCode = 400
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({
              success: false,
              error: 'Missing required fields: url, method, certPem, keyPem'
            }))
            return
          }

          const result = await handleProxyRequest(proxyReq)

          res.statusCode = result.success ? 200 : 502
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify(result))
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error)
          res.statusCode = 400
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({
            success: false,
            error: message,
            durationMs: 0
          }))
        }
      })

      console.log('[provider-proxy] Middleware registered at /api/provider-proxy')
    }
  }
}

export default providerProxyPlugin
