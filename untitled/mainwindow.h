#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "network.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

    void on_processButton_clicked();

    void on_netCheck_stateChanged(int arg1);

    void on_saveButton_clicked();

    void on_confidenceSlider_valueChanged(int value);

    void on_scoreSlider_valueChanged(int value);

    void on_clearButton_clicked();

private:
    Ui::MainWindow *ui;
    Network nw;
    cv::Mat image;
    bool use_net = false;
    float confidence = 0.35;
    float class_threshold_score = 0.2;
    cv::Mat image_processed;
};
#endif // MAINWINDOW_H
