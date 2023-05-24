"""This provides the common docstrings that are used throughout the project."""

import re


class DocstringComponents:
    """Adapted from https://github.com/mwaskom/seaborn/blob/9d8ce6ad4ab213994f0b
    c84d0c46869df7be0b49/seaborn/_docstrings.py."""
    regexp = re.compile(r"\n((\n|.)+)\n\s*", re.MULTILINE)

    def __init__(self, comp_dict, strip_whitespace=True):
        """Read entries from a dict, optionally stripping outer whitespace."""
        if strip_whitespace:
            entries = {}
            for key, val in comp_dict.items():
                m = re.match(self.regexp, val)
                if m is None:
                    entries[key] = val
                else:
                    entries[key] = m.group(1)
        else:
            entries = comp_dict.copy()

        self.entries = entries

    def __getattr__(self, attr):
        """Provide dot access to entries for clean raw docstrings."""
        if attr in self.entries:
            return self.entries[attr]
        try:
            return self.__getattribute__(attr)
        except AttributeError as err:
            # If Python is run with -OO, it will strip docstrings and our lookup
            # from self.entries will fail. We check for __debug__, which is actually
            # set to False by -O (it is True for normal execution).
            # But we only want to see an error when building the docs;
            # not something users should see, so this slight inconsistency is fine.
            if __debug__:
                raise err
            else:
                pass

    @classmethod
    def from_nested_components(cls, **kwargs):
        """Add multiple sub-sets of components."""
        return cls(kwargs, strip_whitespace=False)

    # NOTE Unclear how this will be useful, commenting out for now
    # @classmethod
    # def from_function_params(cls, func):
    #     """Use the numpydoc parser to extract components from existing func."""
    #     params = NumpyDocString(pydoc.getdoc(func))["Parameters"]
    #     comp_dict = {}
    #     for p in params:
    #         name = p.name
    #         type = p.type
    #         desc = "\n    ".join(p.desc)
    #         comp_dict[name] = f"{name} : {type}\n    {desc}"

    #     return cls(comp_dict)


core_args = dict(emitter="""
    emitter (ribs.emitters.EmitterBase): Emitter to use for generating
        solutions and updating the archive.
    """,
                 archive="""
    archive (ribs.archives.ArchiveBase): Archive to use when creating
        and inserting solutions. For instance, this can be
        :class:`ribs.archives.GridArchive`.
    """,
                 solution_batch="""
    solution_batch (numpy.ndarray): Batch of solutions generated by the
        emitter's :meth:`ask()` method.
    """,
                 objective_batch="""
    objective_batch (numpy.ndarray): Batch of objective values.
    """,
                 measures_batch="""
    measures_batch (numpy.ndarray): ``(n, <measure space dimension>)``
        array with the measure space coordinates of each solution.
    """,
                 metadata_batch="""
    metadata_batch (numpy.ndarray): 1D object array containing a metadata
        object for each solution.
    """,
                 status_batch="""
    status_batch (numpy.ndarray): An array of integer statuses
        returned by a series of calls to archive's :meth:`add_single()`
        method or by a single call to archive's :meth:`add()`.
    """,
                 value_batch="""
    value_batch  (numpy.ndarray): 1D array of floats returned by a series of
        calls to archive's :meth:`add_single()` method or by a single call to
        archive's :meth:`add()`. For what these floats represent,
        refer to :meth:`ribs.archives.add()`.
    """,
                 seed="""
    seed (int): Value to seed the random number generator. Set to None to
        avoid a fixed seed.
    """)

core_docs = dict(args=DocstringComponents(core_args))