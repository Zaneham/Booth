{
  description = "BarraCUDA flake (devShell sets LD_LIBRARY_PATH for libhsa-runtime64.so)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;

        rocmPkgs =
          if pkgs ? rocmPackages then pkgs.rocmPackages
          else throw "This nixpkgs does not provide pkgs.rocmPackages; use a newer nixpkgs.";

        # Pick a package that provides libhsa-runtime64.so (ROCR / HSA runtime).
        hsaRuntime =
          let
            candidates = [
              "hsa-rocr"
              "hsa-rocr-runtime"
              "rocm-runtime"
              "rocm-runtime-unwrapped"
            ];
            name = lib.findFirst (n: builtins.hasAttr n rocmPkgs) null candidates;
          in
            if name == null then
              throw "Could not find an ROCm HSA runtime in pkgs.rocmPackages (tried: ${builtins.toString candidates})."
            else
              rocmPkgs.${name};

        barracuda = pkgs.stdenv.mkDerivation {
          pname = "barracuda";
          version = "unstable";
          src = ./.;

          nativeBuildInputs = [ pkgs.gnumake pkgs.makeWrapper ];

          buildPhase = "make";

          installPhase = ''
            runHook preInstall
            install -Dm755 barracuda $out/bin/barracuda
            runHook postInstall
          '';

          # Convenience for `nix run .` so barracuda can dlopen() libhsa-runtime64.so.
          postFixup = ''
            wrapProgram $out/bin/barracuda \
              --prefix LD_LIBRARY_PATH : ${hsaRuntime}/lib:${hsaRuntime}/lib64
          '';

          meta = {
            homepage = "https://github.com/Zaneham/BarraCUDA";
            license = lib.licenses.asl20;
            mainProgram = "barracuda";
            platforms = lib.platforms.linux;
          };
        };
      in
      {
        packages = {
          default = barracuda;
          barracuda = barracuda;
          hsaRuntime = hsaRuntime;
        };

        apps.default = {
          type = "app";
          program = "${barracuda}/bin/barracuda";
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.gcc
            pkgs.gnumake
            hsaRuntime
          ];

          shellHook = ''
            # bc_runtime.c dlopen("libhsa-runtime64.so") relies on the loader search path.
            export LD_LIBRARY_PATH=${hsaRuntime}/lib:${hsaRuntime}/lib64''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          '';
        };
      });
}
