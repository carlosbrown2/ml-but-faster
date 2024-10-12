use linfa::prelude::*;
use linfa_elasticnet::{ElasticNet, Result};
use ndarray::{Array2, Array1};
use pyo3::prelude::*;

#[pyfunction]
fn elastic_net_cv(X: Vec<Vec<f64>>, y: Vec<f64>, penalty: f64, l1_ratio: f64) -> PyResult<Vec<f64>> {
    // Convert Python data (Vec<Vec<f64>>) to ndarray
    let n_samples = X.len();
    let n_features = X[0].len();
    
    let X_ndarray: Array2<f64> = Array2::from_shape_vec((n_samples, n_features), X.into_iter().flatten().collect()).unwrap();
    let y_ndarray: Array1<f64> = Array1::from_vec(y);

    // Split data into training and validation sets
    let dataset = Dataset::new(X_ndarray, y_ndarray);
    let (train, valid) = dataset.split_with_ratio(0.9);

    // Train the Elastic Net model
    let model = ElasticNet::params()
        .penalty(penalty)
        .l1_ratio(l1_ratio)
        .fit(&train)
        .unwrap();

    // Get the coefficients (hyperplane)
    let coefficients = model.hyperplane().to_vec();

    // Optionally, we can also validate on the validation set
    let y_est = model.predict(&valid);
    let r2_score = valid.r2(&y_est).unwrap();

    println!("R² score on validation set: {}", r2_score);
    println!("Intercept: {}", model.intercept());
    
    // Return the coefficients to Python
    Ok(coefficients)
}

#[pymodule]
fn rust_elastic_net(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(elastic_net_cv, m)?)?;
    Ok(())
}
