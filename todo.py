tasks = []
def add_task(task):
	tasks.append(task)
	print(f"Task '{task}' added.")
def view_tasks():
	if tasks:
    	print("\nYour tasks:")
    	for i, task in enumerate(tasks, start=1):
        	print(f"{i}. {task}")
	else:
    	print("No tasks found.")




