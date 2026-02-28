{
  description = "wgpu-llm: LLM inference engine powered by wgpu";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
   };
   nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixvim = {
      url = "github:nix-community/nixvim";
      inputs.nixpkgs.follows = "nixpkgs-unstable";
    };
  };

  outputs = { self, nixpkgs, rust-overlay, nixpkgs-unstable, nixvim }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };
      pkgs-unstable = import nixpkgs-unstable { inherit system; };
      rustToolchain = pkgs.rust-bin.stable.latest.default.override {
        extensions = [ "rust-src" "rust-analyzer" "clippy" ];
        targets = [ "x86_64-pc-windows-gnu"];
      };

      nixvim' = nixvim.legacyPackages.${system};
      nvim = nixvim'.makeNixvim {
        globals.mapleader = " ";
        globals.maplocalleader = " ";

        opts = {
          number = true;
          relativenumber = true;
          signcolumn = "yes";
          termguicolors = true;
          updatetime = 250;
          undofile = true;
          ignorecase = true;
          smartcase = true;
          clipboard = "unnamedplus";
          breakindent = true;
          scrolloff = 10;
          tabstop = 2;
          shiftwidth = 2;
          expandtab = true;
          list = true;
          listchars = "trail:·,nbsp:␣";
        };

        
        autoCmd = [
          {
            event = ["BufReadPost" "BufNewFile"];
            pattern = ["*"];
            command = "highlight ZenkakuSpace ctermbg=red guibg=red | match ZenkakuSpace /\\%u3000/";
          }
          {
            event = ["FocusLost" "BufLeave"];
            pattern = [ "*" ];
            command = "silent! wa";
          }
        ];

        colorschemes.tokyonight = {
          enable = true;
          settings.style = "night";
        };

        plugins = {
          lsp = {
            enable = true;
            servers = {
              rust_analyzer = {
                enable = true;
                installRustc = false;
                installCargo = false;
              };
              wgsl_analyzer.enable = true;
              lua_ls.enable = true;
            };
          };

          blink-cmp = {
            enable = true;
            settings = {
              keymap.preset = "super-tab";
              sources.default = [ "lsp" "path" "buffer" ];
            };
          };

          treesitter = {
            enable = true;
            settings.highlight.enable = true;
            settings.indent.enable = true;
          };

          telescope = {
            enable = true;
            keymaps = {
              "<leader>sf" = { action = "find_files"; options.desc = "Search files"; };
              "<leader>sg" = { action = "live_grep"; options.desc = "Search by grep"; };
              "<leader>sk" = { action = "keymaps"; options.desc = "Search keymaps"; };
              "<leader>sb" = { action = "buffers"; options.desc = "Search buffers"; };
              "<leader>sh" = { action = "help_tags"; options.desc = "Search help"; };
            };
          };

          gitsigns = {
            enable = true;
            settings = {
              on_attach.__raw = ''
                function(bufnr)
                  local gs = require('gitsigns')
                  local map = function(mode, l, r, opts)
                    opts = opts or {}
                    opts.buffer = bufnr
                    vim.keymap.set(mode, l, r, opts)
                  end
                  map('n', ']c', function()
                    if vim.wo.diff then return ']c' end
                    vim.schedule(function() gs.next_hunk() end)
                    return '<Ignore>'
                  end, { expr = true, desc = 'Next hunk' })
                  map('n', '[c', function()
                    if vim.wo.diff then return '[c' end
                    vim.schedule(function() gs.prev_hunk() end)
                    return '<Ignore>'
                  end, { expr = true, desc = 'Previous hunk' })
                end
              '';
            };
          };

          web-devicons.enable = true;

          which-key.enable = true;

          neo-tree = {
            enable = true;
          };

          nvim-autopairs.enable = true;

          indent-blankline.enable = true;

          conform-nvim = {
            enable = true;
            settings = {
              format_on_save = {
                timeout_ms = 500;
                lsp_format = "fallback";
              };
              formatters_by_ft = {
                rust = [ "rustfmt" ];
                nix = [ "nixfmt" ];
                lua = [ "stylua" ];
                "_" = [ "trim_whitespace" ];
              };
            };
          };

          lualine = {
            enable = true;
            settings.options = {
              theme = "tokyonight";
              section_separators = { left = ""; right = ""; };
              component_separators = { left = ""; right = ""; };
            };
          };

          trouble = {
            enable = true;
          };

          lazygit = {
            enable = true;
          };
        };

        keymaps = [
          { mode = "n"; key = "<leader>e"; action = "<cmd>Neotree toggle<cr>"; options.desc = "Toggle file tree"; }
          { mode = "n"; key = "<Esc>"; action = "<cmd>nohlsearch<cr>"; options.desc = "Clear search highlight"; }
          { mode = "n"; key = "<leader>xx"; action = "<cmd>Trouble diagnostics toggle<cr>"; options.desc = "Diagnostics (Trouble)"; }
          { mode = "n"; key = "<leader>xd"; action = "<cmd>Trouble diagnostics toggle filter.buf=0<cr>"; options.desc = "Buffer diagnostics (Trouble)"; }
          { mode = "n"; key = "<leader>gg"; action = "<cmd>LazyGit<cr>"; options.desc = "LazyGit"; }
        ];
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = (with pkgs; [
          rustToolchain
          pkgsCross.mingwW64.stdenv.cc
          pkg-config
          vulkan-headers
          vulkan-loader
          vulkan-tools

          zellij
          ripgrep
          fd
          fzf
          git
          lazygit
        ]) ++ [
          nvim
        ];
        
        shellHook = ''
          echo "wgpu-llm dev enviroment loaded"
        '';
      };
    };
}
