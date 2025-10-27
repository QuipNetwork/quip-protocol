# Quantum Annealing RPC API Specification

## Overview
The QUIP Protocol nodes expose an HTTP RPC endpoint for submitting quantum annealing solve requests. Solve requests are processed by the node's miners and results are stored as transactions in the blockchain.

## Endpoint

**POST** `/solve`

Submits an Ising model problem for quantum annealing and returns solutions.

## Request Format

```json
{
  "h": [array of linear bias coefficients],
  "J": [[i, j, coupling_value], ...] or {key: value},
  "num_samples": integer
}
```

### Parameters

- **h** (required): Array of linear bias coefficients for each variable
- **J** (required): Coupling matrix in one of two formats:
  - List format: `[[i, j, coupling_value], ...]`
  - Dict format: `{"(i,j)": coupling_value, ...}`
- **num_samples** (required): Number of solution samples to return

## Response Format

```json
{
  "samples": [array of solution bitstrings/spin configurations],
  "energies": [array of corresponding energies],
  "transaction_id": "unique identifier",
  "status": "completed"
}
```

### Response Fields

- **samples**: Array of solution configurations (each solution is an array of spin values)
- **energies**: Array of energy values corresponding to each sample
- **transaction_id**: Unique identifier for the transaction (added to blockchain pending pool)
- **status**: Request status ("completed" or error)

## Sample curl Commands

### Example 1: Simple 4-variable Ising problem

```bash
curl -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{
    "h": [0.5, -1.0, 0.3, -0.8],
    "J": [[0, 1, 0.5], [1, 2, -0.3], [2, 3, 0.7]],
    "num_samples": 10
  }'
```

### Example 2: Using dict format for J

```bash
curl -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{
    "h": [0.5, -1.0, 0.3, -0.8],
    "J": {
      "(0,1)": 0.5,
      "(1,2)": -0.3,
      "(2,3)": 0.7
    },
    "num_samples": 10
  }'
```

### Example 3: Larger problem with more samples

```bash
curl -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{
    "h": [-0.5, 0.3, -1.2, 0.8, -0.4, 0.6],
    "J": [
      [0, 1, 0.4],
      [1, 2, -0.5],
      [2, 3, 0.3],
      [3, 4, -0.2],
      [4, 5, 0.6],
      [0, 3, 0.1]
    ],
    "num_samples": 100
  }'
```

## Error Responses

### Missing required fields (400)
```json
{
  "error": "Missing required fields: h, J, num_samples"
}
```

### Invalid J format (400)
```json
{
  "error": "Invalid J format. Must be dict or list."
}
```

### No miners available (503)
```json
{
  "error": "No miners available"
}
```

### Server error (500)
```json
{
  "error": "error description"
}
```

## Transaction Recording

All successful solve requests create a `Transaction` object that is:
1. Added to the node's pending transaction pool
2. Included in the next mined block
3. Permanently recorded on the blockchain

Each transaction contains the full request (h, J, num_samples) and response (samples, energies) data along with a unique transaction ID and timestamp.
