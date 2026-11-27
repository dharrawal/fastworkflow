from typing import Optional
from ...application.todo_list import TodoList
from ...application.todo_manager import TodoListManager

class Context:
    inherits = ["TodoItem"]
    parents = ["TodoList", "TodoListManager"]

    @classmethod
    def get_parent(cls, record: TodoList) -> Optional[TodoListManager]:
        return record.parent 