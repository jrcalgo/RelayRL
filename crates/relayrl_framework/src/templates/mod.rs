//! This module defines traits for training and testing environments in the RelayRL framework.
//! These traits provide a common interface for building observations, running the environment loop,
//! and (in the case of training) calculating performance metrics during model training.

use std::any::Any;

#[derive(Debug, Error, Clone)]
pub enum EnvironmentError {
    #[error("Environment error: {0}")]
    EnvironmentError(String),
}

pub mod environment_traits {
    /// The `EnvironmentTrainingTrait` defines the interface that must be implemented by any
    /// environment where a model is trained. Implementing this trait allows an environment to
    /// interact with the RelayRL framework's training pipeline.
    ///
    /// # Methods
    ///
    /// * `run_environment(&self)` - Executes the main training loop, which includes environment interaction,
    ///   action selection, and model updates.
    /// * `build_observation(&self)` - Constructs an observation from the environment, typically by processing
    ///   raw sensor data or state information.
    /// * `calculate_performance_return(&self)` - Computes a performance metric (e.g., total reward) for the
    ///   current episode or training iteration, which is used to evaluate and improve the model.
    pub trait EnvironmentTrainingTrait {
        fn run_environment(&self) -> Result<(), EnvironmentError>;
        fn build_observation(&self) -> Result<Any, EnvironmentError>;
        fn calculate_performance_return(&self) -> Result<Any, EnvironmentError>;
    }

    /// The `EnvironmentTestingTrait` defines the interface that must be implemented by any
    /// environment where a trained model is evaluated for its inference performance. Implementing
    /// this trait allows an environment to run in a test or inference mode.
    ///
    /// # Methods
    ///
    /// * `run_environment(&self)` - Executes the main inference or testing loop, where the trained model
    ///   is used to interact with the environment and produce actions.
    /// * `build_observation(&self)` - Constructs an observation from the environment, preparing it for
    ///   input into the trained model.
    pub trait EnvironmentTestingTrait {
        fn run_environment(&self) -> Result<(), EnvironmentError>;
        fn build_observation(&self) -> Result<Any, EnvironmentError>;
    }
}

// pub mod network_presets {
//     use crate::prelude::{AgentBuilder, AgentStartParameters, ClientError, RelayRLAgent};
//     use crate::prelude::{
//         InferenceServerStartParameters, RelayRLInferenceServer, RelayRLTrainingServer, ServerBuilder,
//         TrainingServerStartParameters,
//     };
//     pub async fn default_network<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         algorithm_name: String,
//     ) -> (
//         RelayRLAgent<B, D_IN, D_OUT>,
//         RelayRLTrainingServer<B, D_IN, D_OUT>,
//     ) {
//         clientside_inference_shared_training(transport_type, algorithm_name).await
//     }

//     /// Builds and starts a client-side inference agent (1 actor, 1 router) with shared-algorithm on training server and disabled inference server functionality.
//     pub async fn clientside_inference_shared_training<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         algorithm_name: String,
//     ) -> (
//         RelayRLAgent<B, D_IN, D_OUT>,
//         RelayRLTrainingServer<B, D_IN, D_OUT>,
//     ) {
//         match construct_network_architecture(
//             transport_type,
//             ActorInferenceMode::ClientSide,
//             ActorServerMode::Disabled,
//             ActorServerMode::Shared,
//             algorithm_name,
//         )
//         .await
//         {
//             NetworkArchitecture::AgentAndTraining(agent, training_server) => {
//                 (agent, training_server)
//             }
//             _ => panic!("Invalid network architecture"),
//         };
//     }

//     /// Builds and starts a server-side inference agent (1 actor, 1 router) with shared-algorithm on training server.
//     pub async fn shared_serverside_inference_shared_training<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         algorithm_name: String,
//     ) -> (
//         RelayRLAgent<B, D_IN, D_OUT>,
//         RelayRLTrainingServer<B, D_IN, D_OUT>,
//         RelayRLInferenceServer<B, D_IN, D_OUT>,
//     ) {
//         match construct_network_architecture(
//             transport_type,
//             ActorInferenceMode::ServerSide,
//             ActorServerMode::Shared,
//             ActorServerMode::Shared,
//             algorithm_name,
//         )
//         .await
//         {
//             NetworkArchitecture::AgentAndInferenceAndTraining(
//                 agent,
//                 inference_server,
//                 training_server,
//             ) => (agent, training_server, inference_server),
//             _ => panic!("Invalid network architecture"),
//         };
//     }

//     /// Builds and starts a hybrid inference agent (1 actor, 1 router) with shared-algorithm on training server.
//     pub async fn shared_hybrid_inference_shared_training<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         algorithm_name: String,
//     ) -> (
//         RelayRLAgent<B, D_IN, D_OUT>,
//         RelayRLTrainingServer<B, D_IN, D_OUT>,
//         RelayRLInferenceServer<B, D_IN, D_OUT>,
//     ) {
//         match construct_network_architecture(
//             transport_type,
//             ActorInferenceMode::Hybrid,
//             ActorServerMode::Shared,
//             ActorServerMode::Shared,
//             algorithm_name,
//         )
//         .await
//         {
//             NetworkArchitecture::AgentAndInferenceAndTraining(
//                 agent,
//                 inference_server,
//                 training_server,
//             ) => (agent, training_server, inference_server),
//             _ => panic!("Invalid network architecture"),
//         };
//     }

//     /// Builds and starts a client-side inference agent (1 actor, 1 router) with per-actor algorithm on training server and disabled inference server functionality.
//     pub async fn clientside_inference_independent_training<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         algorithm_name: String,
//     ) -> (
//         RelayRLAgent<B, D_IN, D_OUT>,
//         RelayRLTrainingServer<B, D_IN, D_OUT>,
//     ) {
//         match construct_network_architecture(
//             transport_type,
//             ActorInferenceMode::ClientSide,
//             ActorServerMode::Disabled,
//             ActorServerMode::Independent,
//             algorithm_name,
//         )
//         .await
//         {
//             NetworkArchitecture::AgentAndTraining(agent, training_server) => {
//                 (agent, training_server)
//             }
//             _ => panic!("Invalid network architecture"),
//         };
//     }

//     /// Builds and starts a server-side inference agent (1 actor, 1 router) with per-actor algorithm on training server.
//     pub async fn independent_serverside_inference_independent_training<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         algorithm_name: String,
//     ) -> (
//         RelayRLAgent<B, D_IN, D_OUT>,
//         RelayRLTrainingServer<B, D_IN, D_OUT>,
//         RelayRLInferenceServer<B, D_IN, D_OUT>,
//     ) {
//         match construct_network_architecture(
//             transport_type,
//             ActorInferenceMode::ServerSide,
//             ActorServerMode::Independent,
//             ActorServerMode::Independent,
//             algorithm_name,
//         )
//         .await
//         {
//             NetworkArchitecture::AgentAndInferenceAndTraining(
//                 agent,
//                 inference_server,
//                 training_server,
//             ) => (agent, training_server, inference_server),
//             _ => panic!("Invalid network architecture"),
//         };
//     }

//     /// Builds and starts a hybrid inference agent (1 actor, 1 router) with per-actor algorithm on training server.
//     pub async fn independent_hybrid_inference_independent_training<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         algorithm_name: String,
//     ) -> (
//         RelayRLAgent<B, D_IN, D_OUT>,
//         RelayRLTrainingServer<B, D_IN, D_OUT>,
//         RelayRLInferenceServer<B, D_IN, D_OUT>,
//     ) {
//         match construct_network_architecture(
//             transport_type,
//             ActorInferenceMode::Hybrid,
//             ActorServerMode::Independent,
//             ActorServerMode::Independent,
//             algorithm_name,
//         )
//         .await
//         {
//             NetworkArchitecture::AgentAndInferenceAndTraining(
//                 agent,
//                 inference_server,
//                 training_server,
//             ) => (agent, training_server, inference_server),
//             _ => panic!("Invalid network architecture"),
//         };
//     }

//     /// Builds and starts a client-side inference agent (1 actor, 1 router) with disabled training functionality and disabled inference server functionality.
//     pub async fn clientside_inference_disabled_training<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         algorithm_name: String,
//     ) -> RelayRLAgent<B, D_IN, D_OUT> {
//         match construct_network_architecture(
//             transport_type,
//             ActorInferenceMode::ClientSide,
//             ActorServerMode::Disabled,
//             ActorServerMode::Disabled,
//             algorithm_name,
//         )
//         .await
//         {
//             NetworkArchitecture::Agent(agent) => agent,
//             _ => panic!("Invalid network architecture"),
//         };
//     }

//     /// Builds and starts a server-side inference agent (1 actor, 1 router) with disabled training functionality and disabled inference server functionality.
//     pub async fn shared_serverside_inference_disabled_training<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         algorithm_name: String,
//     ) -> (
//         RelayRLAgent<B, D_IN, D_OUT>,
//         RelayRLInferenceServer<B, D_IN, D_OUT>,
//     ) {
//         match construct_network_architecture(
//             transport_type,
//             ActorInferenceMode::ServerSide,
//             ActorServerMode::Shared,
//             ActorServerMode::Disabled,
//             algorithm_name,
//         )
//         .await
//         {
//             NetworkArchitecture::AgentAndInference(agent, inference_server) => {
//                 (agent, inference_server)
//             }
//             _ => panic!("Invalid network architecture"),
//         };
//     }

//     /// Builds and starts a hybrid inference agent (1 actor, 1 router) with disabled training functionality and disabled inference server functionality.
//     pub async fn shared_hybrid_inference_disabled_training<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         algorithm_name: String,
//     ) -> (
//         RelayRLAgent<B, D_IN, D_OUT>,
//         RelayRLInferenceServer<B, D_IN, D_OUT>,
//     ) {
//         match construct_network_architecture(
//             transport_type,
//             ActorInferenceMode::Hybrid,
//             ActorServerMode::Shared,
//             ActorServerMode::Disabled,
//             algorithm_name,
//         )
//         .await
//         {
//             NetworkArchitecture::AgentAndInference(agent, inference_server) => {
//                 (agent, inference_server)
//             }
//             _ => panic!("Invalid network architecture"),
//         };
//     }

//     pub async fn independent_serverside_inference_disabled_training<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         algorithm_name: String,
//     ) -> (
//         RelayRLAgent<B, D_IN, D_OUT>,
//         RelayRLInferenceServer<B, D_IN, D_OUT>,
//     ) {
//         match construct_network_architecture(
//             transport_type,
//             ActorInferenceMode::ServerSide,
//             ActorServerMode::Independent,
//             ActorServerMode::Disabled,
//             algorithm_name,
//         )
//         .await
//         {
//             NetworkArchitecture::AgentAndInference(agent, inference_server) => {
//                 (agent, inference_server)
//             }
//             _ => panic!("Invalid network architecture"),
//         };
//     }

//     pub async fn independent_hybrid_inference_disabled_training<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         algorithm_name: String,
//     ) -> (
//         RelayRLAgent<B, D_IN, D_OUT>,
//         RelayRLInferenceServer<B, D_IN, D_OUT>,
//     ) {
//         match construct_network_architecture(
//             transport_type,
//             ActorInferenceMode::Hybrid,
//             ActorServerMode::Independent,
//             ActorServerMode::Disabled,
//             algorithm_name,
//         )
//         .await
//         {
//             NetworkArchitecture::AgentAndInference(agent, inference_server) => {
//                 (agent, inference_server)
//             }
//             _ => panic!("Invalid network architecture"),
//         };
//     }

//     enum NetworkArchitecture<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     > {
//         Agent(RelayRLAgent<B, D_IN, D_OUT>),
//         AgentAndInference(
//             RelayRLAgent<B, D_IN, D_OUT>,
//             RelayRLInferenceServer<B, D_IN, D_OUT>,
//         ),
//         AgentAndTraining(
//             RelayRLAgent<B, D_IN, D_OUT>,
//             RelayRLTrainingServer<B, D_IN, D_OUT>,
//         ),
//         AgentAndInferenceAndTraining(
//             RelayRLAgent<B, D_IN, D_OUT>,
//             RelayRLInferenceServer<B, D_IN, D_OUT>,
//             RelayRLTrainingServer<B, D_IN, D_OUT>,
//         ),
//     }

//     async fn construct_network_architecture<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         actor_inference_mode: ActorInferenceMode,
//         inference_server_mode: ActorServerMode,
//         training_server_mode: ActorServerMode,
//         algorithm_name: String,
//     ) -> NetworkArchitecture<B, D_IN, D_OUT> {
//         match (inference_server_mode, training_server_mode) {
//             (ActorServerMode::Disabled, ActorServerMode::Disabled) => NetworkArchitecture::Agent(
//                 construct_agent(
//                     transport_type,
//                     actor_inference_mode,
//                     inference_server_mode,
//                     training_server_mode,
//                     algorithm_name,
//                 )
//                 .await,
//             ),
//             (ActorServerMode::Disabled, ActorServerMode::Shared)
//             | (ActorServerMode::Disabled, ActorServerMode::Independent) => {
//                 NetworkArchitecture::AgentAndTraining(
//                     construct_agent(
//                         transport_type,
//                         actor_inference_mode,
//                         inference_server_mode,
//                         training_server_mode,
//                         algorithm_name,
//                     )
//                     .await,
//                     construct_training_server(transport_type).await,
//                 )
//             }
//             (ActorServerMode::Shared, ActorServerMode::Disabled)
//             | (ActorServerMode::Indepdent, ActorServerMode::Disabled) => {
//                 NetworkArchitecture::AgentAndInference(
//                     construct_agent(
//                         transport_type,
//                         actor_inference_mode,
//                         inference_server_mode,
//                         training_server_mode,
//                         algorithm_name,
//                     )
//                     .await,
//                     construct_inference_server(transport_type).await,
//                 )
//             }
//             (ActorSErverMode::Shared, ActorServerMode::Shared)
//             | (ActorServerMode::Independent, ActorServerMode::Independent) => {
//                 NetworkArchitecture::AgentAndInferenceAndTraining(
//                     construct_agent(
//                         transport_type,
//                         actor_inference_mode,
//                         inference_server_mode,
//                         training_server_mode,
//                         algorithm_name,
//                     )
//                     .await,
//                     construct_inference_server(transport_type).await,
//                     construct_training_server(transport_type).await,
//                 )
//             }
//         }
//     }

//     async fn construct_agent<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//         actor_inference_mode: ActorInferenceMode,
//         inference_server_mode: ActorServerMode,
//         training_server_mode: ActorServerMode,
//         algorithm_name: String,
//     ) -> RelayRLAgent<B, D_IN, D_OUT> {
//         let (agent, start_params) = AgentBuilder::<B, D_IN, D_OUT>::builder(
//             transport_type,
//             actor_inference_mode,
//             inference_server_mode,
//             training_server_mode,
//         )
//         .algorithm_name(algorithm_name)
//         .build()
//         .await;
//         agent.start(**start_params).await;
//         agent
//     }

//     async fn construct_training_server<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//     ) -> RelayRLTrainingServer<B, D_IN, D_OUT> {
//         let (server, start_params) =
//             ServerBuilder::<B, D_IN, D_OUT>::builder(transport_type, ServerType::Training);
//         server.start(**start_params).await;
//         server
//     }

//     async fn construct_inference_server<
//         B: Backend + BackendMatcher<Backend = B>,
//         const D_IN: usize,
//         const D_OUT: usize,
//     >(
//         transport_type: TransportType,
//     ) -> RelayRLInferenceServer<B, D_IN, D_OUT> {
//         let (server, start_params) =
//             ServerBuilder::<B, D_IN, D_OUT>::builder(transport_type, ServerType::Inference);
//         server.start(**start_params).await;
//         server
//     }
// }
