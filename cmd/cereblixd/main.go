// cereblixd is the Cereblix full node daemon.
package main

import (
	"flag"
	"log"
	"os"
	"strings"

	"cereblix/core"
	"cereblix/node"
)

func main() {
	var (
		datadir  = flag.String("datadir", "cereblix-data", "data directory")
		p2pAddr  = flag.String("p2p", ":18750", "p2p listen address")
		rpcAddr  = flag.String("rpc", "127.0.0.1:18751", "rpc listen address")
		peers    = flag.String("peers", "http://188.34.181.191:18750", "comma-separated seed peer URLs")
		public   = flag.String("public", "", "publicly reachable URL of this node (advertised to peers)")
		mine     = flag.Bool("mine", false, "enable built-in miner")
		threads  = flag.Int("threads", 2, "miner threads")
		coinbase = flag.String("coinbase", "", "address that receives block rewards")
	)
	flag.Parse()
	log.SetFlags(log.LstdFlags)

	chain, err := core.NewChain(*datadir)
	if err != nil {
		log.Fatalf("chain init: %v", err)
	}
	log.Printf("cereblixd starting | height %d | tip %s", chain.Height(), chain.Tip().Hash()[:16])

	var seeds []string
	for _, p := range strings.Split(*peers, ",") {
		if p = strings.TrimSpace(p); p != "" {
			seeds = append(seeds, p)
		}
	}
	n := node.New(chain, *datadir, *public, seeds)

	if *mine {
		if !core.ValidAddr(*coinbase) {
			log.Println("error: -mine requires a valid -coinbase address (create one with cereblix-wallet new)")
			os.Exit(1)
		}
		n.Mine(*threads, *coinbase)
	}
	log.Fatal(n.Serve(*p2pAddr, *rpcAddr))
}
