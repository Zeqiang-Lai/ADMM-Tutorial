import sys, os
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        ''')

    def setPixmap(self, image):
        super().setPixmap(image)

    def recoverBorder(self):
        self.setStyleSheet('''
                   QLabel{
                       border: 4px dashed #aaa
                   }
               ''')

    def setBorder(self):
        self.setStyleSheet('''
                   QLabel{
                       border: 4px dashed #B22222
                   }
               ''')


class DragDropView(QWidget):
    def __init__(self, action=None):
        super().__init__()
        self.resize(400, 400)
        self.setAcceptDrops(True)

        mainLayout = QVBoxLayout()

        self.photoViewer = ImageLabel()
        mainLayout.addWidget(self.photoViewer)

        self.setLayout(mainLayout)
        self.action = action

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            self.photoViewer.setBorder()
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.photoViewer.recoverBorder()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            self.photoViewer.recoverBorder()
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)
            print(file_path)
            if self.action is not None:
                self.action(file_path)
            event.accept()
        else:
            event.ignore()

    def set_image(self, file_path):
        self.photoViewer.setPixmap(QPixmap(file_path))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = DragDropView()
    demo.show()
    sys.exit(app.exec_())
