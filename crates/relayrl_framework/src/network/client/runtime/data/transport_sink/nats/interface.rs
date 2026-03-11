


pub(crate) struct NatsInterface {
    nats_inference_ops: NatsInferenceOps,
    nats_training_ops: NatsTrainingOps,
    inference_authentication: NatsAuthentication,
    training_authentication: NatsAuthentication,
}

impl<B: Backend + BackendMatcher<Backend = B>> AsyncClientTransportInterface<B> for NatsInterface {
    async fn new(
        client_namespace: Arc<str>,
        shared_client_modes: Arc<ClientModes>,
    ) -> Result<Self, TransportError> {
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> AsyncClientScalingTransportOps<B> for NatsInterface {
    async fn send_client_ids(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        client_ids: Vec<(NamespaceString, ContextString, Uuid)>,
        replace_context: bool,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {

    }

    async fn send_scaling_warning(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        operation: ScalingOperation,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        
    }

    async fn send_scaling_complete(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        operation: ScalingOperation,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        
    }

    async fn send_shutdown_signal(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> AsyncClientInferenceTransportOps<B> for NatsInterface {
    async fn send_inference_model_init_request(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        model_mode: ModelMode,
        model_module: Option<ModelModule<B>>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        
    }

    async fn send_inference_request(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        obs_bytes: Vec<u8>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<RelayRLAction, TransportError> {
        
    }

    async fn send_flag_last_inference(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        reward: f32,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> NatsInferenceExecution for NatsInterface {
    #[inline]
    async fn execute_send_inference_request(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        obs_bytes: &[u8],
        inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError> {
        <NatsInferenceOps as NatsInferenceExecution<B>>::execute_send_inference_request(
            &self.nats_inference_ops,
            actor_entry,
            obs_bytes,
            inference_server_address,
        )
    }

    #[inline]
    async fn execute_send_flag_last_inference(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        reward: &f32,
        inference_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsInferenceOps as NatsInferenceExecution<B>>::execute_send_flag_last_inference(
            &self.nats_inference_ops,
            actor_entry,
            reward,
            inference_server_address,
        )
    }

    #[inline]
    async fn execute_send_inference_model_init_request<MB: Backend + BackendMatcher<Backend = MB>>(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        model_mode: &ModelMode,
        model_module: &Option<ModelModule<MB>>,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsInferenceOps as NatsInferenceExecution<B>>::execute_send_inference_model_init_request(
            &self.nats_inference_ops,
            scaling_entry,
            model_mode,
            model_module,
            inference_scaling_server_address,
        )
    }

    #[inline]
    async fn execute_send_client_ids(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        client_ids: &[(NamespaceString, ContextString, Uuid)],
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsInferenceOps as NatsInferenceExecution<B>>::execute_send_client_ids(
            &self.nats_inference_ops,
            scaling_entry,
            client_ids,
            inference_scaling_server_address,
        )
    }

    #[inline]
    async fn execute_send_scaling_warning(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsInferenceOps as NatsInferenceExecution<B>>::execute_send_scaling_warning(
            &self.nats_inference_ops,
            scaling_entry,
            operation,
            inference_scaling_server_address,
        )
    }

    #[inline]
    async fn execute_send_scaling_complete(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsInferenceOps as NatsInferenceExecution<B>>::execute_send_scaling_complete(
            &self.nats_inference_ops,
            scaling_entry,
            operation,
            inference_scaling_server_address,
        )
    }

    #[inline]
    async fn execute_send_shutdown_signal(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsInferenceOps as NatsInferenceExecution<B>>::execute_send_shutdown_signal(
            &self.nats_inference_ops,
            scaling_entry,
            inference_scaling_server_address,
        )
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> AsyncClientTrainingTransportOps<B> for NatsInterface {
    async fn send_algorithm_init_request(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        actor_entries: Vec<(NamespaceString, ContextString, Uuid)>,
        model_mode: ModelMode,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        
    }

    async fn initial_model_handshake(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        transport_addresses: SharedTransportAddresses,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        
    }

    async fn send_trajectory(
        &self,
        buffer_entry: (NamespaceString, ContextString, Uuid),
        encoded_trajectory: EncodedTrajectory,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        
    }

    async fn listen_for_model(
        &self,
        receiver_entry: (NamespaceString, ContextString, Uuid),
        global_dispatcher_tx: Sender<RoutedMessage>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> NatsTrainingExecution<B> for NatsInterface {
    #[inline]
    async fn execute_listen_for_model(
        &self,
        receiver_entry: &(NamespaceString, ContextString, Uuid),
        global_dispatcher_tx: &Sender<RoutedMessage>,
        model_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_listen_for_model(
            &self.nats_training_ops,
            receiver_entry,
            global_dispatcher_tx,
            model_server_address,
        )
    }

    #[inline]
    async fn execute_send_algorithm_init_request(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        actor_entries: &[(NamespaceString, ContextString, Uuid)],
        model_mode: &ModelMode,
        algorithm: &Algorithm,
        hyperparams: &HashMap<Algorithm, HyperparameterArgs>,
        agent_listener_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_send_algorithm_init_request(
            &self.nats_training_ops,
            scaling_entry,
            actor_entries,
            model_mode,
            algorithm,
            hyperparams,
            agent_listener_address,
        )
    }

    #[inline]
    async fn execute_initial_model_handshake(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_initial_model_handshake(
            &self.nats_training_ops,
            actor_entry,
            agent_listener_address,
        )
    }

    #[inline]
    async fn execute_send_trajectory(
        &self,
        buffer_entry: &(NamespaceString, ContextString, Uuid),
        encoded_trajectory: &EncodedTrajectory,
        trajectory_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_send_trajectory(
            &self.nats_training_ops,
            buffer_entry,
            encoded_trajectory,
            trajectory_server_address,
        )
    }

    #[inline]
    async fn execute_send_client_ids(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        client_ids: &[(NamespaceString, ContextString, Uuid)],
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_send_client_ids(
            &self.nats_training_ops,
            scaling_entry,
            client_ids,
            training_scaling_server_address,
        )
    }

    #[inline]
    async fn execute_send_scaling_warning(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_send_scaling_warning(
            &self.nats_training_ops,
            scaling_entry,
            operation,
            training_scaling_server_address,
        )
    }

    #[inline]
    async fn execute_send_scaling_complete(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_send_scaling_complete(
            &self.nats_training_ops,
            scaling_entry,
            operation,
            training_scaling_server_address,
        )
    }

    #[inline]
    async fn execute_send_shutdown_signal(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_send_shutdown_signal(
            &self.nats_training_ops,
            scaling_entry,
            training_scaling_server_address,
        )
    }
}