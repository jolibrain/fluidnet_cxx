from torch.utils.cpp_extension import load
fluidnet_cpp = load(
        name="fluidnet_cpp",
        sources=[
            "grid.cpp",
            "advect_type.cpp",
            "calc_line_trace.cpp",
            "fluids_init.cpp"
        ],
        verbose=True)
help(fluidnet_cpp)
