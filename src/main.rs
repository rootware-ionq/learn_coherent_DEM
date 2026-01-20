use ndarray::Array2;
use quantrs2_circuit::builder::Circuit;
use quantrs2_core::Register;
use quantrs2_sim::statevector::StateVectorSimulator;

const ANGLE: f64 = 0.01; // the angle for coherent Z errors
const N_QUBITS: usize = 13; // total number of qubits
const N_DATA_Q: usize = 9; // number of data qubits
const N_DETECTOR: usize = N_QUBITS - N_DATA_Q; // number of detector/syndrome qubits

fn main() {
    // Create a circuit with N_QUBITS
    let mut circuit = Circuit::<N_QUBITS>::new();

    build_surface_code_circuit(&mut circuit); // d=3 X-memory rotated surface code

    let simulator = StateVectorSimulator::new();
    let result = circuit.run(simulator).unwrap();

    let (count_arr, count_xor) = get_v_matrices(&result);

    let p_arr = construct_edge_probabilities(&count_arr, &count_xor);
    println!("{}", p_arr);

    // calculate theta
    for i in 0..N_DETECTOR {
        for j in 0..N_DETECTOR {
            if (i == j) || (i == 0 && j == 1) || (i == 1 && j == 2) || (i == 2 && j == 3) {
                let theta = calculate_theta(p_arr[[i, j]]);
                println!("theta_{i}{j} is {}", theta);
            }
        }
    }

    let p_stoch = p_arr[[0, 0]] * 2.0 - 2.0 * p_arr[[0, 0]] * p_arr[[0, 0]];
    let theta_stoch = calculate_theta(p_stoch);
    println!("Original angle is {}", ANGLE);
    println!("Stochastic angle is {}", theta_stoch);
}

/// makes the d=3 X memory rotated surface code circuit
fn build_surface_code_circuit(circuit: &mut Circuit<N_QUBITS>) {
    // put _all_ qubits into |+x> eigenstate
    for i in 0..N_QUBITS {
        circuit.h(i).unwrap();
    }

    // apply an Rz(2\theta) error to all data qubits
    // data qubits are numbered 0 to N_DATA_Q - 1
    for i in 0..N_DATA_Q {
        if i == 6 || i == 5 {
            circuit.rz(i, -2.0 * ANGLE * std::f64::consts::PI).unwrap();
        } else {
            circuit.rz(i, 2.0 * ANGLE * std::f64::consts::PI).unwrap();
        }
    }

    // apply X checks between detector qubits and data qubits
    // detector ancilla qubits are numbered N_DATA_Q to N_QUBITS -1
    // For 13 qubits and 9 data qubits, qubit numbered 9 to 12 are detectors
    circuit.cnot(9, 0).unwrap().cnot(9, 1).unwrap();
    circuit
        .cnot(10, 2)
        .unwrap()
        .cnot(10, 1)
        .unwrap()
        .cnot(10, 4)
        .unwrap()
        .cnot(10, 3)
        .unwrap();
    circuit
        .cnot(11, 5)
        .unwrap()
        .cnot(11, 4)
        .unwrap()
        .cnot(11, 7)
        .unwrap()
        .cnot(11, 6)
        .unwrap();
    circuit.cnot(12, 7).unwrap().cnot(12, 8).unwrap();
    // put detector qubits back into Z basis
    for i in N_DATA_Q..N_QUBITS {
        circuit.h(i).unwrap();
    }
}

/// calculate Rz(2theta) from given edge probability
fn calculate_theta(p: f64) -> f64 {
    1.0 / std::f64::consts::PI * f64::asin(p.sqrt())
}

fn get_v_matrices(result: &Register<N_QUBITS>) -> (Array2<f64>, Array2<f64>) {
    let mut count_arr = Array2::<f64>::zeros((N_DETECTOR, N_DETECTOR));
    let mut count_xor = Array2::<f64>::zeros((N_DETECTOR, N_DETECTOR));

    for (ind, prob) in result.probabilities().iter().enumerate() {
        let bits = format!("{:013b}", ind);
        for i in N_DATA_Q..N_QUBITS {
            for j in N_DATA_Q..N_QUBITS {
                if bits.chars().nth(N_QUBITS - 1 - i).unwrap() == '1'
                    && bits.chars().nth(N_QUBITS - 1 - j).unwrap() == '1'
                {
                    count_arr[[i - N_DATA_Q, j - N_DATA_Q]] += prob;
                }
            }
        }
        for i in N_DATA_Q..N_QUBITS {
            for j in N_DATA_Q..N_QUBITS {
                if bits.chars().nth(N_QUBITS - 1 - i).unwrap() == '1'
                    && bits.chars().nth(N_QUBITS - 1 - j).unwrap() == '0'
                    && i != j
                {
                    count_xor[[i - N_DATA_Q, j - N_DATA_Q]] += prob;
                }
                if bits.chars().nth(N_QUBITS - 1 - i).unwrap() == '0'
                    && bits.chars().nth(N_QUBITS - 1 - j).unwrap() == '1'
                    && i != j
                {
                    count_xor[[i - N_DATA_Q, j - N_DATA_Q]] += prob;
                }
            }
        }
    }

    (count_arr, count_xor)
}

pub fn construct_edge_probabilities(
    count_arr: &Array2<f64>,
    count_xor: &Array2<f64>,
) -> Array2<f64> {
    let mut p_arr = Array2::<f64>::zeros((N_DETECTOR, N_DETECTOR));

    println!("{}", p_arr);

    // calculate p_{ij} for bulk edges first
    for i in 0..N_DETECTOR {
        for j in 0..N_DETECTOR {
            // Not all detectors are connected to each other in the decoding graph
            // these if conditions make it so, if two detectors are connected,
            // we calculate and store their probabilities
            // otherwise they're left as zero.
            if (i == 0 && j == 1)
                || (i == 1 && j == 0)
                || (i == 1 && j == 2)
                || (i == 2 && j == 1)
                || (i == 2 && j == 3)
                || (i == 3 && j == 2)
            {
                // p_arr[[i, j]] = 0.5
                //     - (0.25
                //         - (count_arr[[i, j]] - count_arr[[i, i]] * count_arr[[j, j]])
                //             / (1.0 - 2.0 * (count_arr[[i, i]] + count_arr[[j, j]])
                //                 + 4.0 * count_arr[[i, j]]))
                //     .sqrt();
                p_arr[[i, j]] = 0.5
                    - (0.25
                        - (count_arr[[i, j]] - count_arr[[i, i]] * count_arr[[j, j]])
                            / (1.0 - 2.0 * count_xor[[i, j]]))
                    .sqrt();
            } else {
                p_arr[[i, j]] = 0.0;
            }
        }
    }

    // calculate p_{ij} for boundary edges
    for i in 0..N_DETECTOR {
        let mut denom = 1.0;
        for j in 0..N_DETECTOR {
            // As before, we only include correlations
            // from bulk edges where we have detectors actually connected
            // via an error mechanism
            if (i == 0 && j == 1)
                || (i == 1 && j == 0)
                || (i == 1 && j == 2)
                || (i == 2 && j == 1)
                || (i == 2 && j == 3)
                || (i == 3 && j == 2)
            {
                denom *= 1.0 - 2.0 * p_arr[[i, j]];
            }
        }
        p_arr[[i, i]] = 0.5 + (count_arr[[i, i]] - 0.5) / denom;
    }
    p_arr
}
