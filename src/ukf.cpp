#include "ukf.h"
#include <iostream>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace {

void NormalizeAngleOnComponent(VectorXd& vec, int index) {
  while (vec(index)> M_PI) vec(index) -= 2. * M_PI;
  while (vec(index)<-M_PI) vec(index) += 2. * M_PI;
}

} // namespace
/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.8;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;
  time_us_ = 0;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  Xsig_pred_ = MatrixXd(5, 2 * n_aug_ + 1);

  // set weights 
  weights_.resize(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_) {
    // init timestamp
    time_us_ = meas_package.timestamp_;

    x_.setZero();
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_(0) = meas_package.raw_measurements_(0); // p_x
      x_(1) = meas_package.raw_measurements_(1); // p_y
    }

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_d = meas_package.raw_measurements_(2);
     
      x_(0) = rho * cos(phi);
      x_(1) = rho * sin(phi);
      // To note: this is inaccurate value, aim to initialize velocity which's magnitude/order is close to real value
      x_(2) = rho_d;
    }

    P_.setIdentity();
    P_(2, 2) = 100.;
    P_(3, 3) = 10.;

    is_initialized_ = true;
    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0; // sec

  Prediction(delta_t);   

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }

  time_us_ = meas_package.timestamp_;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // Generate Sigma Points 
  // Get Xsig_aug
  MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);

  VectorXd x_aug(n_aug_);
  x_aug.setZero();
  x_aug.head(n_x_) = x_;

  MatrixXd P_aug(n_aug_, n_aug_);
  P_aug.setZero();
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = pow(std_a_, 2);
  P_aug(n_x_ + 1, n_x_ + 1) = pow(std_yawdd_, 2); 

  Xsig_aug.setZero();
  Xsig_aug.col(0) = x_aug;
  Eigen::MatrixXd A = P_aug.llt().matrixL(); // sqrt of P_aug
  Xsig_aug.block(0, 1, n_aug_, n_aug_) = (sqrt(lambda_ + n_aug_) * A).colwise() + x_aug;
  Xsig_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) = (-sqrt(lambda_ + n_aug_) * A).colwise() + x_aug;

  // Update Xsig_pred_
  VectorXd x_delta(5), x_noise(5);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double p_x = Xsig_aug(0, i); 
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double std_a = Xsig_aug(5, i);
    double std_yawdd = Xsig_aug(6, i); 
    
    x_noise(0) = 0.5 * pow(delta_t, 2) * cos(yaw) * std_a; 
    x_noise(1) = 0.5 * pow(delta_t, 2) * sin(yaw) * std_a;
    x_noise(2) = delta_t * std_a;
    x_noise(3) = 0.5 * pow(delta_t, 2) * std_yawdd;
    x_noise(4) = delta_t * std_yawdd;
    
    if (fabs(yawd) < 0.001) {
      x_delta(0) = v * cos(yaw) * delta_t;
      x_delta(1) = v * sin(yaw) * delta_t;
    }
    else {
      x_delta(0) = v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      x_delta(1) = v / yawd * (-cos(yaw + yawd * delta_t) + cos(yaw));
    }
    x_delta(2) = 0;
    x_delta(3) = yawd * delta_t;
    x_delta(4) = 0; 
    
    Xsig_pred_.col(i) = Xsig_aug.col(i).head(5) + x_delta + x_noise;
  }

  // Get predicted mean and covariance 
  // predict state mean 
  VectorXd x(n_x_);
  x.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x += weights_(i) * Xsig_pred_.col(i);
  }
  // predict state covariance matrix
  MatrixXd P(n_x_, n_x_);
  P.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    NormalizeAngleOnComponent(x_diff, 3);
    P += weights_(i) * x_diff * x_diff.transpose();
  }

  // update x_ and P_
  x_ = x;
  P_ = P;

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  int n_z = 2;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
      Zsig(0, i) = Xsig_pred_(0, i);
      Zsig(1, i) = Xsig_pred_(1, i);
  }

  // calculate mean predicted measurement
  z_pred.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
      z_pred += weights_(i) * Zsig.col(i);
  }
  // calculate innovation covariance matrix S
  S.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
      VectorXd z_diff = Zsig.col(i) - z_pred; 
      S += z_diff * z_diff.transpose() * weights_(i);
  }
  MatrixXd R = MatrixXd::Identity(2, 2); 
  R(0, 0) = pow(std_laspx_, 2);
  R(1, 1) = pow(std_laspy_, 2);
  
  S += R;

  // Update state and covariance 
  // create matrix for cross correlation Tc
  MatrixXd Tc(n_x_, n_z);
  // calculate cross correlation matrix
  Tc.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Tc += weights_(i) * (Xsig_pred_.col(i) - x_) * (Zsig.col(i) - z_pred).transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  // update state mean and covariance matrix
  x_ += K * (z_diff);
  P_ -= K * S * K.transpose(); 

  double nis_lidar = z_diff.transpose() * S.inverse() * z_diff;
  std::cout << "nis_lidar = " << nis_lidar << std::endl;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // Predict measurement 
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
      double px = Xsig_pred_(0, i);
      double py = Xsig_pred_(1, i);
      double v = Xsig_pred_(2, i);
      double yaw = Xsig_pred_(3, i); 
      
      double r = sqrt(pow(px, 2) + pow(py, 2));
      Zsig(0, i) = r; 
      Zsig(1, i) = atan2(py, px); 
      Zsig(2, i) = (v * cos(yaw) * px + v * sin(yaw) * py) / r; 
  }

  // calculate mean predicted measurement
  z_pred.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
      z_pred += weights_(i) * Zsig.col(i);
  }
  // calculate innovation covariance matrix S
  S.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
      VectorXd z_diff = Zsig.col(i) - z_pred; 
      // angle normalization
      NormalizeAngleOnComponent(z_diff, 1);
      S += z_diff * z_diff.transpose() * weights_(i);
  }
  MatrixXd R = MatrixXd::Identity(3, 3); 
  R(0, 0) = pow(std_radr_, 2);
  R(1, 1) = pow(std_radphi_, 2);
  R(2, 2) = pow(std_radrd_, 2); 
  
  S += R;

  // Update state and covariance 
  // create matrix for cross correlation Tc
  MatrixXd Tc(n_x_, n_z);
  // calculate cross correlation matrix
  Tc.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // std::cout << "Zsig.col(" << i << "):" << Zsig.col(i) << std::endl;
    // std::cout << "z_pred:" << z_pred << std::endl;
    // std::cout << "Xsig_pred_.col(" << i << "):" << Xsig_pred_.col(i) << std::endl;
    // std::cout << "x_:" << x_ << std::endl;
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    NormalizeAngleOnComponent(z_diff, 1);

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    NormalizeAngleOnComponent(x_diff, 3);

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  // angle normalization
  NormalizeAngleOnComponent(z_diff, 1);

  // update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose(); 

  double nis_radar = z_diff.transpose() * S.inverse() * z_diff;
  std::cout << "nis_radar = " << nis_radar << std::endl;
}