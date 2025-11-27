// Test tool to compute manifest hash the same way Akash provider does
//
// Usage:
//   go run manifest_hash_test.go '{"manifest":"json"}'
//   echo '{"manifest":"json"}' | go run manifest_hash_test.go
//
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
)

// sortedJSON recursively sorts all map keys and returns canonical JSON
func sortedJSON(v interface{}) interface{} {
	switch val := v.(type) {
	case map[string]interface{}:
		sorted := make(map[string]interface{})
		for k, v := range val {
			sorted[k] = sortedJSON(v)
		}
		return sorted
	case []interface{}:
		result := make([]interface{}, len(val))
		for i, v := range val {
			result[i] = sortedJSON(v)
		}
		return result
	default:
		return v
	}
}

// Marshal with sorted keys (Go's default behavior)
func marshalSorted(v interface{}) ([]byte, error) {
	return json.Marshal(sortedJSON(v))
}

func main() {
	var input string

	if len(os.Args) > 1 {
		input = os.Args[1]
	} else {
		// Read from stdin
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading stdin: %v\n", err)
			os.Exit(1)
		}
		input = string(data)
	}

	fmt.Println("=== Manifest Hash Test ===")
	fmt.Printf("Input length: %d bytes\n", len(input))
	fmt.Printf("Input (first 200): %s\n", truncate(input, 200))

	// Method 1: Direct hash of input (what we're doing in JS)
	hash1 := sha256.Sum256([]byte(input))
	fmt.Printf("\nMethod 1 - Direct SHA256 of input:\n")
	fmt.Printf("  Hash: %s\n", hex.EncodeToString(hash1[:]))

	// Method 2: Parse and re-serialize with Go's json.Marshal (sorts keys by default)
	var parsed interface{}
	if err := json.Unmarshal([]byte(input), &parsed); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing JSON: %v\n", err)
		os.Exit(1)
	}

	remarshaled, err := json.Marshal(parsed)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error re-marshaling: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nMethod 2 - Parse + re-marshal with Go:\n")
	fmt.Printf("  Output length: %d bytes\n", len(remarshaled))
	fmt.Printf("  Output (first 200): %s\n", truncate(string(remarshaled), 200))

	hash2 := sha256.Sum256(remarshaled)
	fmt.Printf("  Hash: %s\n", hex.EncodeToString(hash2[:]))

	// Method 3: Check if Go's marshal differs from input
	fmt.Printf("\nComparison:\n")
	fmt.Printf("  Input == Go output: %v\n", input == string(remarshaled))

	if input != string(remarshaled) {
		fmt.Printf("\n  Differences found! Showing first diff...\n")
		showFirstDiff(input, string(remarshaled))
	}

	// Show sorted keys for debugging
	if arr, ok := parsed.([]interface{}); ok && len(arr) > 0 {
		if obj, ok := arr[0].(map[string]interface{}); ok {
			keys := make([]string, 0, len(obj))
			for k := range obj {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			fmt.Printf("\n  Top-level keys (sorted): %v\n", keys)
		}
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func showFirstDiff(a, b string) {
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	for i := 0; i < minLen; i++ {
		if a[i] != b[i] {
			start := i - 20
			if start < 0 {
				start = 0
			}
			end := i + 20
			if end > minLen {
				end = minLen
			}
			fmt.Printf("  Position %d:\n", i)
			fmt.Printf("    Input:  ...%s...\n", a[start:end])
			fmt.Printf("    Go out: ...%s...\n", b[start:end])
			fmt.Printf("    Input char: %q (0x%02x)\n", a[i], a[i])
			fmt.Printf("    Go char:    %q (0x%02x)\n", b[i], b[i])
			return
		}
	}

	if len(a) != len(b) {
		fmt.Printf("  Lengths differ: input=%d, go=%d\n", len(a), len(b))
	}
}
