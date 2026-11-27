import fastworkflow


class Context:
    """Record is a base of Book and Member: the commands every record has. No object is a
    plain Record, so the workflow never stands on one."""

    occupiable = False

    @classmethod
    def get_state(cls, record):
        """Never called for a live object, because none is a plain Record. Declared deliberately
        ephemeral so the startup check does not list Record as a context that needs a hook."""
        return fastworkflow.EPHEMERAL
