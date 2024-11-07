tasks = []

def add_task(task):
    """Adds a new task to the list"""
    tasks.append(task)
    print(f"Task '{task}' added.")
def view_tasks():
    """Displays all tasks in the list"""
    if tasks:
        print("\nYour tasks:")
        for i, task in enumerate(tasks, start=1):
            print(f"{i}. {task}")
    else:
        print("No tasks found.")

def delete_task(task_number):
    """Deletes a task by its number in the list"""
    if 0 < task_number <= len(tasks):
        removed_task = tasks.pop(task_number - 1)
        print(f"Task '{removed_task}' deleted.")
    else:
        print("Invalid task number.")

def show_menu():
    """Shows the menu of options"""
    print("\nTo-Do List App")
    print("1. Add task")
    print("2. View tasks")
    print("3. Delete task")
    print("4. Exit")

def run():
    """Runs the main loop of the To-Do list app"""
    while True:
        show_menu()
        try:
            choice = int(input("Choose an option (1-4): "))
            if choice == 1:
                task = input("Enter the task to add: ")
                add_task(task)
            elif choice == 2:
                view_tasks()
            elif choice == 3:
                view_tasks()
                try:
                    task_number = int(input("Enter the number of the task to delete: "))
                    delete_task(task_number)
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
            elif choice == 4:
                print("Exiting the app.")
                break
            else:
                print("Invalid option. Please choose a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 4.")

# Run the app
run()
