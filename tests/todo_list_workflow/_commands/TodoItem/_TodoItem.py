from typing import Optional
from ...application.todo_item import TodoItem
from ...application.todo_list import TodoList

class Context:
    parents = ["TodoList"]

    @classmethod
    def get_parent(cls, record: TodoItem) -> Optional[TodoList]:
        return record.parent 