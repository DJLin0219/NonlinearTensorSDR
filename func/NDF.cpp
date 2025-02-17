#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// Helper function to initialize the random seed
//' Initialize Random Seed
//'
//' This function sets the random seed for reproducibility.
//'
//' @param seed An integer value for the random seed.
//' @return None
void initialize_seed(unsigned int seed) {
   Rcpp::Environment base_env("package:base");
   Rcpp::Function set_seed_r = base_env["set.seed"];
   set_seed_r(Rcpp::wrap(seed));
}
 
//' Compute Symmetric Matrix Power
//'
//' This function computes the power of a symmetric matrix.
//'
//' @param a A square symmetric matrix.
//' @param alpha The power exponent.
//' @return The powered matrix.
// [[Rcpp::export]]
 arma::mat matpower(arma::mat a, double alpha) {
   // Ensure the matrix is symmetric
   a = (a + a.t()) / 2;
   
   // Compute eigenvalues and eigenvectors
   arma::vec eigval;
   arma::mat eigvec;
   arma::eig_sym(eigval, eigvec, a);
   
   // Compute the powered matrix
   arma::mat m = eigvec * diagmat(pow(abs(eigval), alpha)) * eigvec.t();
   return m;
 }
 
 //' Compute Trace of a Matrix
 //'
 //' This function computes the trace of a matrix.
 //'
 //' @param a A square matrix.
 //' @return The trace of the matrix.
 // [[Rcpp::export]]
 double tr(const arma::mat& a) {
   return arma::trace(a);
 }
 
 // Helper function to call R's svd function
 //' Perform Singular Value Decomposition (SVD) Using R's `svd` Function
 //'
 //' This function calls R's `svd` function to perform SVD on a matrix.
 //'
 //' @param X A matrix to decompose.
 //' @return A list containing the SVD components: `u`, `d`, and `v`.
 List svdR(const arma::mat &X) {
   Environment base("package:base");
   Function svd = base["svd"];
   List result = svd(X);
   return result;
 }
 
 //' Perform SVD on Multiple Matrices
 //'
 //' This function performs SVD on each column of the input matrix `X` and aligns the results.
 //'
 //' @param X A matrix with dimensions `pl * pr * n`, where `n` is the number of columns.
 //' @param pl The number of rows in each submatrix.
 //' @param pr The number of columns in each submatrix.
 //' @return A list containing:
 //' \item{u}{The left singular vectors.}
 //' \item{v}{The right singular vectors.}
 //' \item{lambda}{The singular values.}
 // [[Rcpp::export]]
 List SVDs(const arma::mat &X, int pl, int pr) {
   // Check if the input dimensions are valid
   if (X.n_rows != pl * pr) {
     stop("The input dimensions pl or pr are incorrect!");
   }
   
   // Initialize matrices to store results
   arma::mat u(pl, 0);
   arma::mat v(pr, 0);
   std::vector<double> lambda;
   
   // Perform SVD for each column of X
   for (size_t i = 0; i < X.n_cols; ++i) {
     // Reshape the column into a pl x pr matrix
     arma::mat Xi = reshape(X.col(i), pl, pr);
     
     // Call R's svd function
     List svd_result = svdR(Xi);
     arma::mat U = as<arma::mat>(svd_result["u"]);
     arma::vec s = as<arma::vec>(svd_result["d"]);
     arma::mat V = as<arma::mat>(svd_result["v"]);
     
     // Align the singular vectors
     double alignment;
     if (i >= 1) {
       alignment = dot(U.col(0), u.col(0));
     } else {
       alignment = 1;
     }
     
     // Ensure consistent orientation of singular vectors
     if (alignment > 0) {
       u = join_horiz(u, U);
       v = join_horiz(v, V);
     } else {
       u = join_horiz(u, -U);
       v = join_horiz(v, -V);
     }
     
     // Store singular values
     lambda.insert(lambda.end(), s.begin(), s.end());
   }
   
   // Return results as a list
   return List::create(
     Named("u") = u,
     Named("v") = v,
     Named("lambda") = lambda
   );
 }
 
 //' Compute Gaussian Kernel Matrix
 //'
 //' This function computes the Gaussian kernel matrix between two sets of points.
 //'
 //' @param x A matrix of input points (n x d).
 //' @param x_new A matrix of new input points (m x d).
 //' @param complexity A complexity parameter for the kernel.
 //' @return The Gaussian kernel matrix (n x m).
 // [[Rcpp::export]]
 arma::mat gram_gauss(const arma::mat& x, const arma::mat& x_new, double complexity) {
   int n = x.n_rows;
   int m = x_new.n_rows;
   
   // Compute pairwise squared Euclidean distances
   arma::mat k2 = x * x.t();
   arma::vec diag_k2 = k2.diag();
   arma::mat k1 = repmat(diag_k2, 1, n).t();
   arma::mat k3 = k1.t();
   arma::mat k = k1 - 2 * k2 + k3;
   
   // Adjust the kernel matrix to ensure positive definiteness
   arma::vec eigval;
   arma::mat eigvec;
   arma::eig_sym(eigval, eigvec, k);
   double epsilon = 1e-6;
   eigval.transform([epsilon](double val) { return std::max(val, epsilon); });
   arma::mat adjusted_k = eigvec * diagmat(eigval) * eigvec.t();
   
   // Compute the bandwidth parameter (sigma) for the Gaussian kernel
   double sigma = accu(sqrt(adjusted_k)) / (2.0 * R::choose(n, 2));
   double gamma = complexity / (2.0 * std::pow(sigma, 2));
   
   // Compute the Gaussian kernel matrix between x and x_new
   arma::mat xmatxmat = x * x.t();
   arma::vec diag_x = xmatxmat.diag();
   arma::mat k_new_1 = repmat(diag_x, 1, m);
   arma::mat k_new_2 = x * x_new.t();
   arma::mat xmatnewxmatnew = x_new * x_new.t();
   arma::vec diag_x_new = xmatnewxmatnew.diag();
   arma::mat k_new_3 = repmat(diag_x_new, 1, n);
   arma::mat result = exp(-gamma * (k_new_1 - 2 * k_new_2 + k_new_3.t()));
   
   return result;
 }
 
 //' Compute Discrete Kernel Matrix
 //'
 //' This function computes the discrete kernel matrix for a vector of labels.
 //'
 //' @param y A vector of labels.
 //' @return The discrete kernel matrix (n x n).
 // [[Rcpp::export]]
 arma::mat gram_dis(const arma::vec& y) {
   int n = y.n_elem;
   arma::mat yy(n, n, fill::zeros);
   
   // Create a matrix where each element is y(i)
   for (int i = 0; i < n; ++i) {
     for (int j = 0; j < n; ++j) {
       yy(i, j) = y(i);
     }
   }
   
   // Compute the discrete kernel matrix
   arma::mat diff = yy - yy.t();
   arma::mat vecker(n, n, fill::zeros);
   vecker.elem(find(diff == 0)).ones();
   vecker.elem(find(diff != 0)).zeros();
   
   return vecker;
 }
 
 //' Compute Linear Kernel Matrix
 //'
 //' This function computes the linear kernel matrix between two matrices.
 //'
 //' @param X A matrix (n x d).
 //' @param Y A matrix (m x d).
 //' @return The linear kernel matrix (n x m).
 // [[Rcpp::export]]
 arma::mat gram_linear(const arma::mat& X, const arma::mat& Y) {
   return X * Y.t();
 }
 
 //' Kernel Generator for Regression Operator
 //'
 //' This function generates kernel matrices for the regression operator.
 //'
 //' @param X A matrix of input data (pl x pr x n).
 //' @param Y A matrix of response data (n x d).
 //' @param pl The number of rows in each submatrix of X.
 //' @param pr The number of columns in each submatrix of X.
 //' @param thre A threshold for convergence (default: 1e-2).
 //' @param iteration The maximum number of iterations (default: 100).
 //' @param kernel_u The kernel type for u ("gaussian" or "linear", default: "gaussian").
 //' @param kernel_v The kernel type for v ("gaussian" or "linear", default: "gaussian").
 //' @param kernel_Y The kernel type for Y ("gaussian" or "discrete", default: "gaussian").
 //' @return A list containing:
 //' \item{A}{The regression operator matrix.}
 //' \item{Gy}{The kernel matrix for Y.}
 //' \item{Gu}{The kernel matrix for u.}
 //' \item{Gv}{The kernel matrix for v.}
 // [[Rcpp::export]]
 List KernelGenerator(const arma::mat& X, const arma::mat& Y, int pl, int pr,
                      double thre = 1e-2, int iteration = 100, std::string kernel_u = "gaussian",
                      std::string kernel_v = "gaussian", std::string kernel_Y = "gaussian") {
   int n = X.n_cols;
   int r = std::min(pl, pr);
   
   // Perform SVD on X
   List svdr = SVDs(X, pl, pr);
   arma::mat u = as<arma::mat>(svdr["u"]);
   arma::mat v = as<arma::mat>(svdr["v"]);
   arma::vec lambda = as<arma::vec>(svdr["lambda"]);
   
   // Compute kernel matrices
   arma::mat Gu, Gv, Gy;
   
   if (kernel_u == "gaussian") {
     Gu = gram_gauss(trans(u), trans(u), 1.0);
   }
   
   if (kernel_v == "gaussian") {
     Gv = gram_gauss(trans(v), trans(v), 1.0);
   }
   
   if (kernel_Y == "gaussian") {
     if (Y.n_cols == 1) {
       Gy = gram_gauss(reshape(Y, n, 1), reshape(Y, n, 1), 1.0);
     } else {
       Gy = gram_gauss(Y, Y, 1.0);
     }
     arma::mat Q = eye<arma::mat>(n, n) - ones<arma::mat>(n, n) / n;
     Gy = Q * Gy * Q;
   } else if (kernel_Y == "discrete") {
     arma::vec Y_vec = vectorise(Y);
     Gy = gram_dis(Y_vec);
     arma::mat Q = eye<arma::mat>(n, n) - ones<arma::mat>(n, n) / n;
     Gy = Q * Gy * Q;
   }
   
   // Compute the regression operator matrix A
   arma::mat A(r * n, n, fill::zeros);
   for (int i = 0; i < n; ++i) {
     A.submat(i * r, i, (i + 1) * r - 1, i) = lambda.subvec(i * r, (i + 1) * r - 1);
   }
   arma::vec a = mean(A, 1);
   for (int j = 0; j < n; ++j) {
     A.col(j) -= a;
   }
   
   return List::create(
     Named("A") = A,
     Named("Gy") = Gy,
     Named("Gu") = Gu,
     Named("Gv") = Gv
   );
 }
 
 //' Generalized Cross-Validation (GCV) for X
 //'
 //' This function computes the Generalized Cross-Validation (GCV) value for X.
 //'
 //' @param As A matrix (n x n).
 //' @param GY A matrix (n x n).
 //' @param epsilon_x A regularization parameter.
 //' @return The GCV value for X.
 // [[Rcpp::export]]
 double gcvx(const arma::mat& As, const arma::mat& GY, double epsilon_x) {
   // Compute As^T * As
   arma::mat AstAs = As * As.t();
   
   // Compute maximum eigenvalue of As^T * As
   arma::vec eigval;
   arma::mat eigvec;
   eig_sym(eigval, eigvec, AstAs);
   double lam = eigval(0);
   
   // Compute Tar matrix
   arma::mat Tar = AstAs * matpower(AstAs + epsilon_x * lam * eye<mat>(As.n_rows, As.n_rows), -1);
   
   // Compute nu
   arma::mat AsGY = As * GY;
   double nu = norm(AsGY - Tar * AsGY, "fro");
   nu *= nu;
   
   // Compute de
   arma::mat identity = eye<mat>(As.n_rows, As.n_rows);
   double de = tr(identity - Tar);
   de *= de;
   
   // Compute and return GCVX value
   return nu / de;
 }
 
 //' Compute Optimal Epsilon X
 //'
 //' This function computes the optimal regularization parameter epsilon_x using GCV.
 //'
 //' @param A A matrix (n x n).
 //' @param Gy A matrix (n x n).
 //' @param epsilon_x A vector of candidate regularization parameters.
 //' @return The optimal epsilon_x value.
 // [[Rcpp::export]]
 double compute_epsilon_x(const arma::mat& A, const arma::mat& Gy, const arma::vec& epsilon_x) {
   int n = epsilon_x.n_elem;
   arma::vec gcv_x(n, fill::zeros);
   
   // Compute GCVX values for each epsilon_x
   for (int i = 0; i < n; ++i) {
     gcv_x(i) = gcvx(A, Gy, epsilon_x(i));
   }
   
   // Find index of minimum GCVX value
   int min_index = gcv_x.index_min();
   
   // Return epsilon_x corresponding to minimum GCVX value
   return epsilon_x(min_index);
 }
 
 //' Generalized Cross-Validation (GCV) for U and V
 //'
 //' This function computes the Generalized Cross-Validation (GCV) value for U and V.
 //'
 //' @param As A matrix (n x n).
 //' @param Gu A matrix (n x n).
 //' @param Gv A matrix (n x n).
 //' @param GY A matrix (n x n).
 //' @param epsilon_x A regularization parameter for X.
 //' @param epsilon_u A regularization parameter for U.
 //' @param epsilon_v A regularization parameter for V.
 //' @return The GCV value for U and V.
 // [[Rcpp::export]]
 double gcvuv(const arma::mat& As, const arma::mat& Gu, const arma::mat& Gv,
              const arma::mat& GY, double epsilon_x, double epsilon_u, double epsilon_v) {
   // Compute maximum eigenvalues
   double lam_u = eig_sym(Gu).max();
   double lam_v = eig_sym(Gv).max();
   double lam_x = eig_sym(As * As.t()).max();
   
   // Compute GuGuinv and GvGvinv matrices
   arma::mat GuGuinv = Gu * matpower(Gu + epsilon_u * lam_u * eye<mat>(Gu.n_rows, Gu.n_cols), -1);
   arma::mat GvGvinv = Gv * matpower(Gv + epsilon_v * lam_v * eye<mat>(Gv.n_rows, Gv.n_cols), -1);
   
   // Compute M matrix
   arma::mat M = matpower(As * As.t() + epsilon_x * lam_x * eye<mat>(As.n_rows, As.n_rows), -1) * As * GY;
   
   // Compute nu
   double nu = 0;
   for (int i = 0; i < GY.n_rows; ++i) {
     arma::mat ei = zeros<mat>(GY.n_rows, 1);
     ei(i, 0) = 1;
     nu += pow(norm(GuGuinv * diagmat(vectorise(M * ei)) * GvGvinv - diagmat(vectorise(M * ei)), "fro"), 2);
   }
   
   // Compute de
   double de = pow(pow(As.n_rows, 2) - tr(GuGuinv) * tr(GvGvinv), 2);
   
   // Compute and return GCVUV value
   return nu / de;
 }
 
 //' Compute Optimal Epsilon U and V
 //'
 //' This function computes the optimal regularization parameters epsilon_u and epsilon_v using GCV.
 //'
 //' @param As A matrix (n x n).
 //' @param Gu A matrix (n x n).
 //' @param Gv A matrix (n x n).
 //' @param GY A matrix (n x n).
 //' @param epsilon_x A regularization parameter for X.
 //' @param epsilon_u A vector of candidate regularization parameters for U.
 //' @param epsilon_v A vector of candidate regularization parameters for V.
 //' @return A list containing the optimal epsilon_u and epsilon_v values.
 // [[Rcpp::export]]
 List compute_epsilon_uv(const arma::mat& As, const arma::mat& Gu, const arma::mat& Gv,
                         const arma::mat& GY, double epsilon_x, const arma::vec& epsilon_u,
                         const arma::vec& epsilon_v) {
   int n_u = epsilon_u.n_elem;
   int n_v = epsilon_v.n_elem;
   
   arma::mat gcv_uv(n_u, n_v, fill::zeros);
   
   // Compute GCVUV values for each combination of epsilon_u and epsilon_v
   for (int i = 0; i < n_u; ++i) {
     for (int j = 0; j < n_v; ++j) {
       gcv_uv(i, j) = gcvuv(As, Gu, Gv, GY, epsilon_x, epsilon_u(i), epsilon_v(j));
     }
   }
   
   // Find indices of minimum GCVUV value
   arma::uword min_index_u, min_index_v;
   gcv_uv.min(min_index_u, min_index_v);
   
   // Return list containing minimum epsilon_u and epsilon_v
   return List::create(Named("epsilon_u") = epsilon_u(min_index_u),
                       Named("epsilon_v") = epsilon_v(min_index_v));
 }
 

 
 //' NSPGSIRTu: Nonlinear Sufficient Dimension Reduction with Tucker Decomposition
 //'
 //' @param X Predictor matrix (p x n).
 //' @param Y Response matrix (d x n).
 //' @param pl Left dimension of the predictor.
 //' @param pr Right dimension of the predictor.
 //' @param dl Target left reduced dimension.
 //' @param dr Target right reduced dimension.
 //' @param threshold Convergence threshold (default: 1e-2).
 //' @param max_iter Maximum number of iterations (default: 100).
 //' @param kernel_u Kernel type for left singular space (default: "gaussian").
 //' @param kernel_v Kernel type for right singular space (default: "gaussian").
 //' @param kernel_Y Kernel type for response variable (default: "gaussian").
 //' @param epsilon_u Regularization parameter for left kernel matrix (default: 1e-1).
 //' @param epsilon_v Regularization parameter for right kernel matrix (default: 1e-1).
 //' @param epsilon_x Regularization parameter for X (default: 1e-2).
 //' @return A list containing estimated matrices `f`, `g`, and `h`.
 // [[Rcpp::export]]
 List NSPGSIRTu(const arma::mat& X, const arma::mat& Y, int pl, int pr, int dl, int dr, double thre = 1e-2, int max_iter = 100,
              std::string kernel_u = "gaussian", std::string kernel_v = "gaussian", std::string kernel_Y = "gaussian",
              double epsilon_u = 1e-1, double epsilon_v = 1e-1, double epsilon_x = 1e-2) {
   int n = X.n_cols;
   int r = std::min(pl, pr);
   
   // SVD
   List svdr = SVDs(X, pl, pr);
   arma::mat u = as<arma::mat>(svdr["u"]);
   arma::mat v = as<arma::mat>(svdr["v"]);
   arma::vec lambda = as<arma::vec>(svdr["lambda"]);
   
   // Kernel Generator
   List kern = KernelGenerator(X, Y, pl, pr, thre, max_iter, kernel_u, kernel_v, kernel_Y);
   arma::mat Gu = as<arma::mat>(kern["Gu"]);
   arma::mat Gv = as<arma::mat>(kern["Gv"]);
   arma::mat Gy = as<arma::mat>(kern["Gy"]);
   arma::mat A = as<arma::mat>(kern["A"]);
   
   double lam_u = eig_sym(Gu).max();
   double lam_v = eig_sym(Gv).max();
   double lam_x = eig_sym(A * A.t()).max();
   
   arma::mat Guinv = matpower(Gu + epsilon_u * lam_u * arma::eye<arma::mat>(Gu.n_rows, Gu.n_cols), -1);
   arma::mat Gvinv = matpower(Gv + epsilon_v * lam_v * arma::eye<arma::mat>(Gv.n_rows, Gv.n_cols), -1);
   arma::mat AtAinv = matpower(A * A.t() + epsilon_x * lam_x * arma::eye<arma::mat>(A.n_rows, A.n_rows), -1);
   
   arma::mat R1 = zeros<arma::mat>(n * r, n);
   arma::mat AtAinvAGy = AtAinv * A * Gy;
   List R;
   for (int i = 0; i < n; ++i) {
     arma::mat ei = zeros<arma::mat>(Gy.n_rows, 1);
     ei(i, 0) = 1;
     // arma::mat R_temp = Guinv * diagmat(vectorise(AtAinvAGy * ei)) *Gvinv;
     arma::mat R_temp = diagmat(vectorise(AtAinvAGy * ei));
     // arma::mat R_temp =  diagmat(vectorise(AtAinvAGy * ei));
     R.push_back(R_temp);
   }
   
   arma::mat f0 = randn<arma::mat>(n * r, dl);
   arma::mat g = zeros<arma::mat>(n * r, dr);
   f0 /= norm(f0, "fro");
   arma::mat h0 = randn<arma::mat>(dl * dr, n);
   h0 /= norm(h0, "fro");
   arma::mat f  = zeros<arma::mat>(n * r, dl);
   f= f0;
   arma::mat h = zeros<arma::mat>(dl, dr);
   h = h0;
   double err = 0;
   double erro = 1e8;
   
   arma::mat Guinvinv = matpower(Guinv, -1);
   arma::mat Gvinvinv = matpower(Gvinv, -1);
   
   for (int iter = 0; iter < max_iter; ++iter) {
     // Step III: Compute g
     arma::mat tHFH = zeros<arma::mat>(dr, dr);
     arma::mat RHf = zeros<arma::mat>(n * r, dr);
     arma::mat tfGuf = zeros<arma::mat>(dl, dl);
     tfGuf = f.t() * Gu * f;
     arma::mat GuinvGuf = zeros<arma::mat>(n*r, dl);
     GuinvGuf = Guinv * Gu * f;
     //arma::mat Guf = zeros<arma::mat>(n*r, dl);
     //Guf = Gu * f;
     
     for (int i = 0; i < n; ++i) {
       arma::mat R_i = R[i];
       arma::mat h_mat = reshape(h.col(i), dl, dr);
       tHFH += (h_mat.t() * tfGuf * h_mat);
       RHf += (Gvinv * (R_i * (GuinvGuf * h_mat)));
       //RHf += (R_i.t() * (Guf * h_mat));
     }
     g = RHf * reshape(matpower(tHFH, -1),dr,dr);
     
     // Step IV: Compute f
     arma::mat HGtH = zeros<arma::mat>(dl, dl);
     arma::mat RHg = zeros<arma::mat>(n * r, dl);
     arma::mat tgGvg = zeros<arma::mat>(dr, dr);
     tgGvg = g.t() * Gv * g;
     arma::mat GvinvGvg = zeros<arma::mat>(n*r, dr);
     GvinvGvg = Gvinv * Gv * g;
     //arma::mat Gvg = zeros<arma::mat>(n*r, dr);
     //Gvg = Gv * g;
     
     for (int i = 0; i < n; ++i) {
       arma::mat R_i = R[i];
       arma::mat h_mat = reshape(h.col(i), dl, dr);
       HGtH += (h_mat * tgGvg * h_mat.t());
       RHg += (Guinv * (R_i * (GvinvGvg * h_mat.t())));
       //RHg += (R_i * (Gvg * h_mat.t()));
     }
     f = RHg * reshape(matpower(HGtH, -1),dl,dl);
     
     // Step V: Compute h
     arma::mat fGufinv = reshape(matpower((f.t() * Gu * f), -1),dl,dl);
     arma::mat gGvginv = reshape(matpower(g.t() * Gv * g, -1),dr,dr);
     
     
     for (int i = 0; i < n; ++i) {
       arma::mat R_i = R[i];
       //arma::mat t0 = f.t() * Guinvinv* R_i * Gvinvinv* g;
       arma::mat t0 = f.t() * R_i * g;
       h.col(i) = arma::vectorise(fGufinv * t0 * gGvginv);
     }
     
     // Compute error
     err = 0;
     
     for (int i = 0; i < n; ++i) {
       arma::mat R_i = R[i];
       arma::mat E =  R_i - (Guinvinv * f) * arma::reshape(h.col(i), dl, dr) * (g.t() * Gvinvinv);
       //arma::mat E = R_i - (Guinvinv * f) * arma::reshape(h.col(i), dl, dr) * (g.t() * Gvinvinv);
       err += norm(E, "fro") / ((n * n * n) * (r * r));
     }
     
     if (err > (1 - thre) * erro) {
       break;
     }
     
     erro = err;
   }
   
   // Call R's svd function
   List svd_result_F = svdR(reshape(f, n * r, dl));
   arma::mat Fo_u = as<arma::mat>(svd_result_F["u"]);
   arma::vec Fo_d = as<arma::vec>(svd_result_F["d"]);
   arma::mat Fo_v = as<arma::mat>(svd_result_F["v"]);
   f = Fo_u;
   
   List svd_result_G = svdR(reshape(g, n * r, dr));
   arma::mat Go_u = as<arma::mat>(svd_result_G["u"]);
   arma::vec Go_d = as<arma::vec>(svd_result_G["d"]);
   arma::mat Go_v = as<arma::mat>(svd_result_G["v"]);
   g = Go_u;
   
   for (int iii = 0; iii < n; ++iii) {
     if (dr != 1 && dl != 1) {
       h.col(iii) = vectorise(diagmat(Fo_d) * Fo_v * reshape(h.col(iii), dl, dr) * Go_v.t() * diagmat(Go_d));
     } else if (dr == 1 && dl != 1) {
       h.col(iii) = vectorise(diagmat(Fo_d) * Fo_v * reshape(h.col(iii), dl, dr) * (Go_v * Go_d[0]));
     } else if (dr != 1 && dl == 1) {
       h.col(iii) = vectorise((Fo_v * Fo_d[0]) * reshape(h.col(iii), dl, dr) * Go_v.t() * diagmat(Go_d));
     } else if (dr == 1 && dl == 1) {
       h.col(iii) = vectorise((Fo_v * Fo_d[0]) * reshape(h.col(iii), dl, dr) * (Go_v * Go_d[0]));
     }
   }
   
   return List::create(Named("f") = f,
                       Named("g") = g,
                       Named("h") = h,
                       Named("error") = err,
                       Named("iter") = max_iter,
                       Named("erro") = erro);
 }


 //' NSPGSIRII_utils: Helper Function for Nonlinear Sufficient Dimension Reduction
 //'
 //' This function performs an iterative estimation of transformation matrices `f`, `g`, and `h` 
 //' for a dimension reduction model using precomputed kernel matrices.
 //'
 //' @param X Predictor matrix (p x n).
 //' @param pl Left dimension of the predictor.
 //' @param pr Right dimension of the predictor.
 //' @param Gu Left kernel matrix (n x n).
 //' @param Gv Right kernel matrix (n x n).
 //' @param Gu_inv Inverse of Gu (n x n).
 //' @param Gv_inv Inverse of Gv (n x n).
 //' @param M List of precomputed matrices (size n).
 //' @param max_iter Maximum number of iterations (default: 100).
 //' @param threshold Convergence threshold (default: 1e-2).
 //' @return A list containing estimated matrices `f`, `g`, and `h`.
 // [[Rcpp::export]]
 List NSPGSIRII_utils(const arma::mat& X, int pl, int pr, 
                       const arma::mat& Gu, const arma::mat& Gv, 
                       const arma::mat& Gu_inv, const arma::mat& Gv_inv,
                       const List& M, int max_iter = 100, double threshold = 1e-2) {
   
   // Get sample size and effective rank
   int n = X.n_cols;
   int r = std::min(pl, pr);
   int dl = 1, dr = 1;  // Target dimension reduction parameters
   
   // Initialize transformation matrices
   arma::mat f = randu<arma::mat>(n * r, dl);
   arma::mat g = zeros<arma::mat>(n * r, dr);
   arma::mat h = randu<arma::mat>(dl * dr, n);
   
   // Normalize initial matrices
   f /= norm(f, "fro");
   h /= norm(h, "fro");
   
   // Compute inverse of `Gu_inv` and `Gv_inv` for efficiency
   arma::mat Gu_inv_inv = matpower(Gu_inv, -1);
   arma::mat Gv_inv_inv = matpower(Gv_inv, -1);
   
   double diff = 0.0;  // Convergence tracking variable
   
   // Iterative optimization process
   for (int iter = 0; iter < max_iter; ++iter) {
     
     // Step 1: Update `g`
     arma::mat tHFH = zeros<arma::mat>(dr, dr);
     arma::mat RHf = zeros<arma::mat>(n * r, dr);
     arma::mat tfGuf = f.t() * Gu * f;
     arma::mat Guf = Gu * f;
     
     for (int i = 0; i < n; ++i) {
       arma::mat M_i = M[i];
       arma::mat h_mat = reshape(h.col(i), dl, dr);
       tHFH += h_mat.t() * tfGuf * h_mat;
       RHf += M_i.t() * (Guf * h_mat);
     }
     arma::mat g_new = RHf * reshape(matpower(tHFH, -1), dr, dr);
     
     // Step 2: Update `f`
     arma::mat HGtH = zeros<arma::mat>(dl, dl);
     arma::mat RHg = zeros<arma::mat>(n * r, dl);
     arma::mat tgGvg = g_new.t() * Gv * g_new;
     arma::mat Gvg = Gv * g_new;
     
     for (int i = 0; i < n; ++i) {
       arma::mat M_i = M[i];
       arma::mat h_mat = reshape(h.col(i), dl, dr);
       HGtH += h_mat * tgGvg * h_mat.t();
       RHg += M_i * (Gvg * h_mat.t());
     }
     arma::mat f_new = RHg * reshape(matpower(HGtH, -1), dl, dl);
     
     // Step 3: Update `h`
     arma::mat fGuf_inv = reshape(matpower(f_new.t() * Gu * f_new, -1), dl, dl);
     arma::mat gGvg_inv = reshape(matpower(g_new.t() * Gv * g_new, -1), dr, dr);
     arma::mat h_new = zeros<arma::mat>(dl * dr, n);
     
     for (int i = 0; i < n; ++i) {
       arma::mat M_i = M[i];
       arma::mat t0 = f_new.t() * Gu.t() * M_i * Gv * g_new;
       h_new.col(i) = arma::vectorise(fGuf_inv * t0 * gGvg_inv);
     }
     
     // Step 4: Compute convergence error
     diff = norm(f - f_new) + norm(g - g_new) + norm(h - h_new);
     
     // Update matrices
     f = f_new;
     g = g_new;
     h = h_new;
     
     // Check convergence
     if (diff < threshold) {
       break;
     }
   }
   
   // Return the optimized matrices
   return List::create(Named("f") = f,
                       Named("g") = g,
                       Named("h") = h);
 }
 
 //' Converts a matrix to a symmetric matrix
 //'
 //' This function ensures that a given matrix is symmetric by averaging it with its transpose.
 //'
 //' @param M Input matrix (n x n).
 //' @return A symmetric matrix.
 // [[Rcpp::export]]
 arma::mat to_symmetric(const arma::mat& M) {
   return (M + M.t()) / 2.0;
 }

 //' NSPGSIRCP: Nonlinear Sufficient Dimension Reduction with CP Decomposition
 //'
 //' This function estimates the sufficient dimension reduction (SDR) space using 
 //' nonlinear kernelized sufficient dimension reduction via projected slices.
 //'
 //' @param X Predictor matrix (p x n).
 //' @param Y Response matrix (d_y x n).
 //' @param pl Left dimension of the predictor tensor.
 //' @param pr Right dimension of the predictor tensor.
 //' @param d Target dimension for sufficient dimension reduction.
 //' @param threshold Convergence threshold for stopping criterion (default: 1e-2).
 //' @param max_iter Maximum number of iterations (default: 100).
 //' @param kernel_u Kernel type for left predictor kernel (default: "gaussian").
 //' @param kernel_v Kernel type for right predictor kernel (default: "gaussian").
 //' @param kernel_Y Kernel type for response variable (default: "gaussian").
 //' @param epsilon_u Regularization parameter for left kernel matrix (default: 1e-1).
 //' @param epsilon_v Regularization parameter for right kernel matrix (default: 1e-1).
 //' @param epsilon_x Regularization parameter for response kernel matrix (default: 1e-2).
 //'
 //' @return A list containing:
 //'   - `f`: Estimated left transformation matrix (nr x d).
 //'   - `g`: Estimated right transformation matrix (nr x d).
 //'   - `h`: Estimated response transformation matrix (d x n).
 //'
 // [[Rcpp::export]]
 List NSPGSIRCP(const arma::mat &X, const arma::mat &Y, int pl, int pr, int d, 
                double threshold = 1e-2, int max_iter = 100, 
                std::string kernel_u = "gaussian", std::string kernel_v = "gaussian",
                std::string kernel_Y = "gaussian", double epsilon_u = 1e-1, 
                double epsilon_v = 1e-1, double epsilon_x = 1e-2) {
   
   // Define dimensions
   int n = X.n_cols;  // Sample size
   int r = std::min(pl, pr);  // Rank for decomposition
   
   // Step 1: Compute Singular Value Decomposition (SVD)
   List svdr = SVDs(X, pl, pr);
   arma::mat u = as<arma::mat>(svdr["u"]);
   arma::mat v = as<arma::mat>(svdr["v"]);
   arma::vec lambda = as<arma::vec>(svdr["lambda"]);
   
   // Step 2: Generate Kernel Matrices
   List kern = KernelGenerator(X, Y, pl, pr, threshold, max_iter, kernel_u, kernel_v, kernel_Y);
   arma::mat Gu = as<arma::mat>(kern["Gu"]);  // Left kernel
   arma::mat Gv = as<arma::mat>(kern["Gv"]);  // Right kernel
   arma::mat Gy = as<arma::mat>(kern["Gy"]);  // Response kernel
   arma::mat A = as<arma::mat>(kern["A"]);    // Transformation matrix
   
   // Step 3: Compute Regularized Inverses for Stability
   double lam_u = eig_sym((Gu + Gu.t()) / 2).max();
   double lam_x = eig_sym(A * A.t()).max();
   double lam_v = eig_sym((Gv + Gv.t()) / 2).max();
   
   arma::mat Gu_inv = matpower(Gu + epsilon_u * lam_u * arma::eye<arma::mat>(Gu.n_rows, Gu.n_cols), -1);
   arma::mat Gv_inv = matpower(Gv + epsilon_v * lam_v * arma::eye<arma::mat>(Gv.n_rows, Gv.n_cols), -1);
   arma::mat AtA_inv = matpower(A * A.t() + epsilon_x * lam_x * arma::eye<arma::mat>(A.n_rows, A.n_rows), -1);
   
   // Step 4: Compute Residual Matrices
   arma::mat R1 = zeros<arma::mat>(n * r, n);
   arma::mat AtAinvAGy = AtA_inv * A * Gy;
   List R;
   
   for (int i = 0; i < n; ++i) {
     arma::mat ei = zeros<arma::mat>(Gy.n_rows, 1);
     ei(i, 0) = 1;
     arma::mat R_temp = Gu_inv * diagmat(vectorise(AtAinvAGy * ei)) * Gv_inv;
     R.push_back(R_temp);
   }
   
   // Clone residuals for iterative updates
   List M = clone(R);
   
   // Step 5: Initialize Transformation Matrices
   arma::mat f(n * r, d, fill::zeros);  // Left transformation matrix
   arma::mat g(n * r, d, fill::zeros);  // Right transformation matrix
   arma::mat h(d, n, fill::zeros);      // Response transformation matrix
   
   // Step 6: Iterative Optimization
   for (int dim = 0; dim < d; dim++) {
     // Solve NSPGSIRII using the precomputed residuals
     Rcpp::List result = NSPGSIRII_utils(X, pl, pr, Gu, Gv, Gu_inv, Gv_inv, M, max_iter, threshold);
     
     // Extract transformation matrices
     f.col(dim) = arma::reshape(Rcpp::as<arma::mat>(result["f"]), n * r, 1);
     g.col(dim) = arma::reshape(Rcpp::as<arma::mat>(result["g"]), n * r, 1);
     h.row(dim) = arma::reshape(Rcpp::as<arma::mat>(result["h"]), 1, n);
     
     // If we have computed all dimensions, break early
     if (dim == d - 1) {
       break;
     }
     
     // Step 7: Update Residual Matrices
     for (int i = 0; i < n; ++i) {
       arma::mat M_i = Rcpp::as<arma::mat>(M[i]);
       M[i] = M_i - f.col(dim) * h(dim, i) * (g.col(dim).t());
     }
   }
   
   // Return estimated transformation matrices
   return List::create(Named("f") = f, 
                       Named("g") = g, 
                       Named("h") = h);
 }
 
 //' NSPGSIR_predict_Tu: Prediction using NSPGSIR with Tucker Decomposition
 //'
 //' Computes the projected predictor matrix for new data using the estimated 
 //' transformation matrices from NSPGSIR.
 //'
 //' @param X Training predictor matrix (p x n).
 //' @param Y Training response matrix (d_y x n).
 //' @param X_new New predictor matrix (p x n_new).
 //' @param pl Left dimension of the predictor tensor.
 //' @param pr Right dimension of the predictor tensor.
 //' @param dl Left dimension of the transformed space.
 //' @param dr Right dimension of the transformed space.
 //' @param f Estimated left transformation matrix.
 //' @param g Estimated right transformation matrix.
 //'
 //' @return The transformed new predictor matrix (dl * dr x n_new).
 //'
 //' @export
 // [[Rcpp::export]]
 arma::mat NSPGSIR_predict_Tu(const arma::mat& X, const arma::mat& Y, const arma::mat& X_new, 
                           int pl, int pr, int dl, int dr, const arma::mat& f, const arma::mat& g) {
   // Perform SVD for training data 
   List res = SVDs(X, pl, pr);
   arma::mat u = as<arma::mat>(res["u"]);
   arma::mat v = as<arma::mat>(res["v"]);
   arma::vec lambda = as<arma::vec>(res["lambda"]);
   
   // Perform SVD for test data
   List res_test = SVDs(X_new, pl, pr);
   arma::mat u_t = as<arma::mat>(res_test["u"]);
   arma::mat v_t = as<arma::mat>(res_test["v"]);
   arma::vec lambda_t = as<arma::vec>(res_test["lambda"]);
   
   int n = X.n_cols;
   int n_t = X_new.n_cols;
   int r = std::min(pl, pr);
   arma::mat X_test_d = zeros<arma::mat>(dl * dr, n_t);
   
   for (int k = 0; k < n_t; ++k) {
     arma::mat Gunew = gram_gauss(u.t(), u_t.t(), 1);
     arma::mat Gvnew = gram_gauss(v.t(), v_t.t(), 1);
     arma::vec lambda_slice = lambda_t.subvec(k * r, (k + 1) * r - 1);
     arma::mat lam = repmat(lambda_slice, 1, n*r);
     arma::mat temp = f.t() * (lam % Gunew.cols(k * r, (k + 1) * r - 1).t()).t() * Gvnew.cols(k * r, (k + 1) * r - 1).t() * g;
     X_test_d.col(k) = vectorise(temp);
   }
   return X_test_d;
 }
 
 
 //' NSPGSIR_predict_CP: Prediction using NSPGSIR with CP Decomposition
 //'
 //' Computes the projected predictor matrix for new data using the estimated 
 //' transformation matrices from NSPGSIR with a CANDECOMP/PARAFAC (CP) decomposition approach.
 //'
 //' @param X Training predictor matrix (p x n).
 //' @param Y Training response matrix (d_y x n).
 //' @param X_new New predictor matrix (p x n_new).
 //' @param pl Left dimension of the predictor tensor.
 //' @param pr Right dimension of the predictor tensor.
 //' @param d Target dimension for sufficient dimension reduction.
 //' @param f Estimated left transformation matrix (nr x d).
 //' @param g Estimated right transformation matrix (nr x d).
 //'
 //' @return The transformed new predictor matrix (d x n_new).
 //'
 //' @export
 // [[Rcpp::export]]
 arma::mat NSPGSIR_predict_CP(const arma::mat& X, const arma::mat& Y, const arma::mat& X_new, 
                              int pl, int pr, int d, const arma::mat& f, const arma::mat& g) {
   // Perform SVD for training data 
   List res = SVDs(X, pl, pr);
   arma::mat u = as<arma::mat>(res["u"]);
   arma::mat v = as<arma::mat>(res["v"]);
   arma::vec lambda = as<arma::vec>(res["lambda"]);
   
   // Perform SVD for test data
   List res_test = SVDs(X_new, pl, pr);
   arma::mat u_t = as<arma::mat>(res_test["u"]);
   arma::mat v_t = as<arma::mat>(res_test["v"]);
   arma::vec lambda_t = as<arma::vec>(res_test["lambda"]);
   
   int n = X.n_cols;
   int n_t = X_new.n_cols;
   int r = std::min(pl, pr);
   arma::mat X_test_d = zeros<arma::mat>(d, n_t);
   double gamma_u = 1; // Kernel parameter for left transformation
   double gamma_v = 1; // Kernel parameter for right transformation
   
   for (int k = 0; k < n_t; ++k) {
     // Compute kernel Gram matrices for new data
     arma::mat Gunew = gram_gauss(u.t(), u_t.t(), gamma_u);
     arma::mat Gvnew = gram_gauss(v.t(), v_t.t(), gamma_v);
     
     // Extract corresponding lambda slice
     arma::vec lambda_slice = lambda_t.subvec(k * r, (k + 1) * r - 1);
     arma::mat lam = arma::repmat(lambda_slice, 1, n * r);
     
     // Compute transformed prediction
     arma::vec temp(d);
     for (int i = 0; i < d; ++i) {
       temp(i) = arma::as_scalar(f.col(i).t() * 
         (lam % Gunew.cols(k * r, (k + 1) * r - 1).t()).t() * 
         Gvnew.cols(k * r, (k + 1) * r - 1).t() * g.col(i));
     }
     X_test_d.col(k) = temp;
   }
   
   return X_test_d;
 }
 
 