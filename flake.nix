{
  description = "A Nix-flake-based Python development environment with pre-commit shell hook";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    git-hooks-nix.inputs.nixpkgs.follows = "nixpkgs";
    git-hooks-nix.url = "github:cachix/git-hooks.nix";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs @ {
    self,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      imports = [
        inputs.git-hooks-nix.flakeModule
      ];

      # Supported systems
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem = {
        config,
        pkgs,
        ...
      }: let
        /*
        Change this value ({major}.{min}) to update the Python
        virtual-environment version. If you bump this, delete
        the `.venv` directory so the hook can rebuild it for
        the new version. Then reload the dev shell.
        */
        version = "3.12";

        concatMajorMinor = v:
          pkgs.lib.pipe v [
            pkgs.lib.versions.splitVersion
            (pkgs.lib.sublist 0 2)
            pkgs.lib.concatStrings
          ];

        python = pkgs."python${concatMajorMinor version}";
      in {
        # Pre-commit hooks configuration
        pre-commit.settings = {
          # Add any hooks you want here; tools listed here are made available
          # in the dev shell via `enabledPackages`.
          hooks = {
            alejandra.enable = true;
            ruff.enable = true;
            # black.enable = true;  # uncomment if you want black as well
          };
        };

        # Dev shell with shellHook installing pre-commit
        devShells.default = pkgs.mkShell {
          shellHook = ''
            ${config.pre-commit.installationScript}
            echo 1>&2 "Welcome to the development shell (Python ${version})!"
          '';

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc pkgs.zlib];

          # Tools needed for your workflow, plus anything required by hooks.
          packages =
            config.pre-commit.settings.enabledPackages
            ++ [
              python
              # Add whatever else you'd like here.
              # pkgs.basedpyright
              # python.pkgs.black
            ]
            ++ (
              with pkgs; [uv]
            );
        };
      };

      flake = {};
    };
}
