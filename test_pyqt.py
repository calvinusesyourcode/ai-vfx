from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QRadioButton, QPushButton, QLabel
import sys

def show_choice():
    if radio1.isChecked():
        label.setText("You selected Option 1")
    elif radio2.isChecked():
        label.setText("You selected Option 2")
    elif radio3.isChecked():
        label.setText("You selected Option 3")

app = QApplication(sys.argv)

# Create a QWidget as the main window
window = QWidget()
window.setWindowTitle("PyQt5 Multiple Choice Menu")

# Create a QVBoxLayout to manage layout
layout = QVBoxLayout()

# Create radio buttons
radio1 = QRadioButton("Option 1")
radio2 = QRadioButton("Option 2")
radio3 = QRadioButton("Option 3")

# Add radio buttons to layout
layout.addWidget(radio1)
layout.addWidget(radio2)
layout.addWidget(radio3)

# Create a button to show the selected choice
button = QPushButton("Show Choice")
button.clicked.connect(show_choice)
layout.addWidget(button)

# Create a label to display the choice
label = QLabel("Your choice will be displayed here")
layout.addWidget(label)

# Set the layout for the window
window.setLayout(layout)

# Show the window
window.show()

# Start the event loop
sys.exit(app.exec_())
