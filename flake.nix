{
  description = "wgpu-llm: LLM inference engine powered by wgpu";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
   };
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      system = "aarch64-darwin";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };
      rustToolchain = pkgs.rust-bin.stable.latest.default.override {
        extensions = [ "rust-src" "rust-analyzer" "clippy" ];
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          rustToolchain
          pkg-config

          ripgrep
          fd
          fzf
          git
          lazygit
        ];
        
        shellHook = ''
          echo "wgpu-llm dev enviroment loaded"
        '';
      };
    };
}
