#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "network.h"
#include <QMessageBox>
#include <QFileDialog>
#include "opencv_functional_v2.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow), nw("..\\classes.txt", "..\\yolov5s.onnx")
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open image"), QDir::currentPath(), tr("Image Files (*.jpg *.jpeg *.png)"));
    if (!fileName.isEmpty()) {
        image = cv::imread(fileName.toStdString());
        ui->imageLabel->setPixmap(QPixmap::fromImage(QImage(image.data, image.cols, image.rows, image.step, QImage::Format_BGR888)));
        ui->imageProcessedLabel->clear();
        image_processed = cv::Mat();
    }
}


void MainWindow::on_processButton_clicked()
{
    if (image.empty()) {
        QMessageBox::warning(this, tr("Invalid Image"), tr("No Image"));
    } else {
        image_processed = image.clone();
        if (use_net) {
           nw.GetDetection(image_processed, confidence, class_threshold_score);
        } else {
            processImage(image_processed);
        }
         ui->imageProcessedLabel->setPixmap(QPixmap::fromImage(QImage(image_processed.data, image_processed.cols, image_processed.rows, image_processed.step, QImage::Format_BGR888)));
    }
}


void MainWindow::on_netCheck_stateChanged(int arg1)
{
    use_net = arg1;
}


void MainWindow::on_saveButton_clicked()
{
    if (image_processed.empty()) {
       QMessageBox::warning(this, tr("Can't save to file!"), tr("No image processeed or empty filename"));
       return;
    }
 QString filename =  QFileDialog::getSaveFileName(this,
                                                  tr("Save Address Book"), "",
                                                  tr("Image file (*.png *.jpg *.jpeg);;All Files (*)"));

 if (!filename.isEmpty()) {
     cv::imwrite(filename.toStdString(), image_processed);
 }
}


void MainWindow::on_confidenceSlider_valueChanged(int value)
{
    confidence = static_cast<double>(value) / 100;
    ui->confidenceLabel->setText(QString::number(confidence));
}


void MainWindow::on_scoreSlider_valueChanged(int value)
{
    class_threshold_score = static_cast<double>(value) / 100;
    ui->scoreLabel->setText(QString::number(class_threshold_score));
}


void MainWindow::on_clearButton_clicked()
{
    ui->imageLabel->clear();
    ui->imageProcessedLabel->clear();
    image_processed = cv::Mat();
    image = cv::Mat();
}

