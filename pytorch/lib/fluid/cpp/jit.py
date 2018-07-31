from torch.utils.cpp_extension import load
advection_cpp = load(
        name="advection_cpp",
        sources=[
            "grid.cpp",
            "advect_type.cpp",
            "calc_line_trace.cpp",
            "advection.cpp"
        ],
        verbose=True)
help(advection_cpp)
