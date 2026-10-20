{
  description = "HPFL のローカル smoke 用 Python 環境";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.cudaSupport = false;
      };
      commonPackages = ps: with ps; [
        numpy
        scipy
        scikit-learn
        pillow
        pyyaml
        yacs
        wandb
        psutil
        setproctitle
        matplotlib
        pandas
        easydict
        pytest
        h5py
        tqdm
      ];
      python = pkgs.python312.withPackages (ps: [ ps.torch ps.torchvision ] ++ commonPackages ps);
      gpuPkgs = import nixpkgs {
        inherit system;
        config = {
          cudaSupport = false;
          allowUnfreePredicate = pkg:
            let name = nixpkgs.lib.getName pkg;
            in builtins.elem name [ "torch" "triton" "nccl-binary" ]
              || builtins.match "(cuda|libcu|libnv|cudnn).*" name != null;
        };
      };
      # 公式 wheel 内の共有ライブラリだけを使用し、NCCL をコンパイルしない。
      ncclBinary = gpuPkgs.stdenvNoCC.mkDerivation {
        pname = "nccl-binary";
        version = "2.26.2";
        src = gpuPkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/67/ca/f42388aed0fddd64ade7493dbba36e1f534d4e6fdbdd355c6a90030ae028/nvidia_nccl_cu12-2.26.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl";
          sha256 = "694cf3879a206553cc9d7dbda76b13efaf610fdb70a50cba303de1b0d1530ac6";
        };
        nativeBuildInputs = [ gpuPkgs.unzip gpuPkgs.autoPatchelfHook ];
        buildInputs = [ gpuPkgs.stdenv.cc.cc.lib ];
        unpackPhase = ''
          runHook preUnpack
          unzip -q "$src"
          runHook postUnpack
        '';
        dontBuild = true;
        dontStrip = true;
        installPhase = ''
          runHook preInstall
          mkdir -p "$out/lib" "$out/include"
          cp -a nvidia/nccl/lib/libnccl.so.2 "$out/lib/"
          ln -s libnccl.so.2 "$out/lib/libnccl.so"
          cp -a nvidia/nccl/include/. "$out/include/"
          runHook postInstall
        '';
        meta = {
          license = gpuPkgs.lib.licenses.bsd3;
          sourceProvenance = [ gpuPkgs.lib.sourceTypes.binaryNativeCode ];
          platforms = [ system ];
        };
      };
      gpuTorch = gpuPkgs.python312Packages.torch-bin.override {
        cudaPackages = gpuPkgs.cudaPackages // { nccl = ncclBinary; };
      };
      gpuVision = gpuPkgs.python312Packages.torchvision-bin.override { torch-bin = gpuTorch; };
      gpuPython = gpuPkgs.python312.withPackages (ps: [ gpuTorch gpuVision ] ++ commonPackages ps);
      shellHook = ''
        export PYTHONNOUSERSITE=1
        export WANDB_MODE=disabled
        export WANDB_DISABLED=true
      '';
    in {
      devShells.${system} = {
        default = pkgs.mkShell {
          packages = [ python pkgs.git pkgs.coreutils ];
          inherit shellHook;
        };
        gpu = gpuPkgs.mkShell {
          packages = [ gpuPython gpuPkgs.git gpuPkgs.coreutils ];
          shellHook = shellHook + ''
            export LD_LIBRARY_PATH=/usr/lib/wsl/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            export CUBLAS_WORKSPACE_CONFIG=:4096:8
          '';
        };
      };
      packages.${system} = {
        default = python;
        gpu = gpuPython;
        nccl-binary = ncclBinary;
      };
    };
}
