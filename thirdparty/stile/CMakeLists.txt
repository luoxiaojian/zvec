# stile evaluates forward predicates per candidate. It links zvec's own Arrow
# targets, so the binary carries one Arrow, and takes RE2 from Arrow's
# bundled dependencies.
set(STILE_ARROW_PROVIDER TARGETS CACHE STRING "" FORCE)
set(STILE_ARROW_TARGETS "Arrow::arrow_compute;Arrow::arrow_static"
    CACHE STRING "" FORCE)
set(STILE_RE2_TARGET Arrow::arrow_depends CACHE STRING "" FORCE)
set(STILE_RE2 ON CACHE STRING "" FORCE)
add_subdirectory(stile stile EXCLUDE_FROM_ALL)
