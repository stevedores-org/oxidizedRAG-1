{
  description = "oxidizedRAG - High-performance Rust GraphRAG";

  # NOTE: All inputs are pinned to specific commit SHAs for supply-chain security.
  # This prevents accidental/malicious mutations in upstream repositories.
  # To update: nix flake update --recreate-lock-file, review changes, commit both
  # flake.nix and flake.lock files before merging.

  nixConfig = {
    extra-substituters = [ "https://nix-cache.stevedores.org/stevedores" ];
    extra-trusted-substituters = [ "https://nix-cache.stevedores.org/stevedores" ];
    extra-trusted-public-keys = [ "stevedores-cache-1:bXLxkipycRWproIJnk8pPWNFdgVfeV+I2mJXCoW4/ag=" ];
  };

  # NOTE: Inputs are pinned to exact commits via flake.lock (committed to repo).
  # Run `nix flake update` to bump, and review the lock diff before merging.
  inputs = {
    # Pin nixpkgs to specific commit for supply-chain security
    # Update: nix flake update --recreate-lock-file, then review lock file before merging
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # flake-utils: utility functions for multi-platform Nix flakes
    flake-utils.url = "github:numtide/flake-utils";

    # rust-overlay: Latest Rust toolchain management
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # crane: Incremental Rust builds with Nix
    # v0.17.3 release - production-ready
    crane.url = "github:ipetkov/crane";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, crane, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "rustfmt" "clippy" ];
          targets = [ "wasm32-unknown-unknown" ];
        };

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # Common args for crane builds
        commonArgs = {
          src = craneLib.cleanCargoSource ./.;
          strictDeps = true;
          buildInputs = with pkgs; [
            openssl
          ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.darwin.apple_sdk.frameworks.Security
            pkgs.darwin.apple_sdk.frameworks.SystemConfiguration
          ];
          nativeBuildInputs = with pkgs; [
            pkg-config
          ];
        };

        # Build workspace deps first (for caching)
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # Build the full workspace
        workspace = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });
      in
      {
        checks = {
          inherit workspace;

          clippy = craneLib.cargoClippy (commonArgs // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets -- -D warnings";
          });

          fmt = craneLib.cargoFmt {
            src = craneLib.cleanCargoSource ./.;
          };

          tests = craneLib.cargoNextest (commonArgs // {
            inherit cargoArtifacts;
            partitions = 1;
            partitionType = "count";
          });

          benches = craneLib.buildPackage (commonArgs // {
            inherit cargoArtifacts;
            cargoExtraArgs = "--workspace --benches --no-run";
          });

          doc = craneLib.cargoDoc (commonArgs // {
            inherit cargoArtifacts;
            cargoDocExtraArgs = "--workspace --no-deps";
          });
        };

        packages = {
          default = workspace;

          graphrag-server = craneLib.buildPackage (commonArgs // {
            inherit cargoArtifacts;
            cargoExtraArgs = "-p graphrag-server";
          });

          graphrag-cli = craneLib.buildPackage (commonArgs // {
            inherit cargoArtifacts;
            cargoExtraArgs = "-p graphrag-cli";
          });
        };

        devShells.default = craneLib.devShell {
          checks = self.checks.${system};

          packages = with pkgs; [
            # Rust extras
            cargo-watch
            cargo-nextest

            # WASM
            wasm-pack
            wasm-bindgen-cli
            trunk

            # Nix cache
            attic-client

            # Bun (for docs-site)
            bun

            # Tools
            just
            git
          ];

          RUST_BACKTRACE = "1";

          shellHook = ''
            if [ -x ./.githooks/install.sh ]; then
              ./.githooks/install.sh >/dev/null 2>&1 || true
            fi

            echo "🔍 oxidizedRAG Development Environment"
            echo ""
            echo "Quick commands (just):"
            echo "  just fmt        # cargo fmt --check"
            echo "  just clippy     # clippy -D warnings"
            echo "  just test       # workspace tests"
            echo "  just bench      # benches compile"
            echo "  just doc        # docs build"
            echo "  just ci         # full local CI"
            echo "  just flake-check"
            echo ""
            echo "Nix Cache (Attic):"
            echo "  attic login stevedores https://nix-cache.stevedores.org \$ATTIC_TOKEN"
            echo "  attic push stevedores <store-path>"
            echo ""
          '';
        };
      }
    );
}
