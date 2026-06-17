import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import sys


class GoalInstructionPublisher(Node):

    def __init__(self):
        super().__init__('goal_instruction_publisher')

        self.publisher_ = self.create_publisher(String, '/goal_instruction', 10)
        self.get_logger().info("Nodo avviato. Scrivi il goal e premi INVIO.")
        
        self.input_thread = threading.Thread(target=self.read_input_loop, daemon=True)
        self.input_thread.start()

    def read_input_loop(self):
        while rclpy.ok():
            try:
                sys.stdout.write("Goal > ")
                sys.stdout.flush()
                text = sys.stdin.readline()
                if text is None or text == '':
                    break
                text = text.strip()
                if text == "":
                    continue

                msg = String()
                msg.data = text
                self.publisher_.publish(msg)
                self.get_logger().info(f"Pubblicato goal: '{text}'")

            except Exception as e:
                self.get_logger().error(f"Errore input: {str(e)}")
                break


def main(args=None):
    rclpy.init(args=args)
    node = GoalInstructionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Chiusura nodo...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()