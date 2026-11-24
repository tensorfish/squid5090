const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    // Default to .fast (ReleaseFast) so plain `zig build` produces a full-speed
    // inference binary (TigerStyle's 243 release-active assertions remain enabled).
    const optimize: std.builtin.Optimize = b.option(
        std.builtin.Optimize,
        "optimize",
        "Optimization mode: fast (default), safe, small, debug",
    ) orelse switch (b.graph.release_mode) {
        .off, .any, .fast => .fast,
        .safe => .safe,
        .small => .small,
    };

    const cuda_arch = b.option(
        []const u8,
        "cuda-arch",
        "CUDA Blackwell SM architecture (default: 120a)",
    ) orelse "120a";

    const default_cuda_path: []const u8 = if (@import("builtin").os.tag == .linux and
        std.os.linux.access("/usr/local/cuda/bin/nvcc", 0) != 0 and
        std.os.linux.access("/opt/cuda/bin/nvcc", 0) == 0)
        "/opt/cuda"
    else
        "/usr/local/cuda";

    const cuda_path = b.option(
        []const u8,
        "cuda-path",
        "CUDA toolkit root directory (auto-detects /usr/local/cuda or /opt/cuda)",
    ) orelse default_cuda_path;

    const nvcc_bin = b.fmt("{s}/bin/nvcc", .{cuda_path});
    const cuda_lib = b.fmt("{s}/lib64", .{cuda_path});
    const gencode = b.fmt("arch=compute_{s},code=sm_{s}", .{ cuda_arch, cuda_arch });

    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Compile all 5 Blackwell CUDA translation units in parallel with full
    // content-addressed Zig caching (including header dependency tracking).
    const cuda_headers = [_][]const u8{
        "cuda/config.hpp",
        "cuda/weights.hpp",
        "cuda/kernels.hpp",
        "cuda/kernels_common.cuh",
    };

    const cuda_sources = [_]struct { src: []const u8, obj: []const u8 }{
        .{ .src = "cuda/kernels_gemv.cu", .obj = "kernels_gemv.o" },
        .{ .src = "cuda/kernels_inkling.cu", .obj = "kernels_inkling.o" },
        .{ .src = "cuda/kernels_dspark.cu", .obj = "kernels_dspark.o" },
        .{ .src = "cuda/kernels_prefill.cu", .obj = "kernels_prefill.o" },
        .{ .src = "cuda/cuda_abi.cu", .obj = "cuda_abi.o" },
    };

    for (cuda_sources) |cu| {
        const nvcc = b.addSystemCommand(&.{
            nvcc_bin,
            "-O3",
            "--use_fast_math",
            "-std=c++20",
            "-gencode",
            gencode,
            "--expt-relaxed-constexpr",
            "-I",
        });
        nvcc.addDirectoryArg(b.path("cuda"));
        nvcc.addArg("-c");
        nvcc.addFileArg(b.path(cu.src));
        nvcc.addArg("-o");
        const obj_file = nvcc.addOutputFileArg(cu.obj);
        for (cuda_headers) |hdr| {
            nvcc.addFileInput(b.path(hdr));
        }
        root_module.addObjectFile(obj_file);
    }

    root_module.addLibraryPath(.{ .cwd_relative = cuda_lib });
    root_module.addRPath(.{ .cwd_relative = cuda_lib });
    root_module.linkSystemLibrary("cudart", .{});
    root_module.linkSystemLibrary("cublas", .{});
    root_module.linkSystemLibrary("stdc++", .{});
    root_module.linkSystemLibrary("m", .{});
    root_module.linkSystemLibrary("pthread", .{});

    const exe = b.addExecutable(.{
        .name = "squid5090",
        .root_module = root_module,
    });
    b.installArtifact(exe);

    // `zig build run -- <subcommand> <args...>`
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run squid5090 with arbitrary CLI arguments");
    run_step.dependOn(&run_cmd.step);

    // `zig build serve -- <model_dir> [--port 8000 --ctx 65536 --mtp 7 ...]`
    const serve_cmd = b.addRunArtifact(exe);
    serve_cmd.step.dependOn(b.getInstallStep());
    serve_cmd.addArg("serve");
    serve_cmd.addPassthruArgs();
    const serve_step = b.step("serve", "Start the native Zig OpenAI HTTP/SSE API server");
    serve_step.dependOn(&serve_cmd.step);

    // `zig build bench -- <model_dir> [--mode nvfp4 --mtp 7 ...]`
    const bench_cmd = b.addRunArtifact(exe);
    bench_cmd.step.dependOn(b.getInstallStep());
    bench_cmd.addArg("bench");
    bench_cmd.addPassthruArgs();
    const bench_step = b.step("bench", "Run multi-iteration median decode benchmark");
    bench_step.dependOn(&bench_cmd.step);

    // `zig build longctx -- <model_dir> [--mode nvfp4 --kv q4 --ctx 1048576]`
    const longctx_cmd = b.addRunArtifact(exe);
    longctx_cmd.step.dependOn(b.getInstallStep());
    longctx_cmd.addArg("longctx");
    longctx_cmd.addPassthruArgs();
    const longctx_step = b.step("longctx", "Run long-context scaling sweep (up to 1M tokens)");
    longctx_step.dependOn(&longctx_cmd.step);
}
