use crate::environment::{decode_action_confidence, decode_action_index};
use crate::openttd::{GameSubsystem, OpenTtdBridge, OpenTtdBridgeError, OpenTtdConfig};
use crate::subsystems::{
    ACTOR_SPECS, SubsystemActorSpec, command_from_action, f32_slice_to_bytes, project_observation,
    reward_for_transition,
};
use crate::{ACTION_DIM, OBSERVATION_DIM};
use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor, TensorData as BurnTensorData};
use relayrl::network::{RelayRLAgent, RelayRLAgentActors};
use relayrl::types::tensor::relayrl::DeviceType;
use std::collections::BTreeMap;
use std::error::Error;

type ActorUuid = uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ActorBinding {
    pub id: ActorUuid,
    pub spec: SubsystemActorSpec,
    pub last_reward: f32,
}

#[derive(Debug, Clone)]
pub struct CrossRuntimeSignal {
    pub source_actor: &'static str,
    pub subsystem: GameSubsystem,
    pub action_index: usize,
    pub command_description: &'static str,
    pub reward_after_world_step: f32,
}

pub struct OpenTtdHost {
    bridge: OpenTtdBridge,
    config: OpenTtdConfig,
    actor_bindings: Vec<ActorBinding>,
    last_signals: Vec<CrossRuntimeSignal>,
}

impl OpenTtdHost {
    pub fn new(config: OpenTtdConfig) -> Result<Self, OpenTtdBridgeError> {
        Ok(Self {
            bridge: OpenTtdBridge::new(config.clone())?,
            config,
            actor_bindings: Vec::new(),
            last_signals: Vec::new(),
        })
    }

    pub fn actor_bindings(&self) -> &[ActorBinding] {
        &self.actor_bindings
    }

    pub fn last_signals(&self) -> &[CrossRuntimeSignal] {
        &self.last_signals
    }

    pub async fn create_actor_graph(
        &mut self,
        agent: &mut RelayRLAgent<NdArray>,
        max_traj_length: usize,
    ) -> Result<(), Box<dyn Error>> {
        self.actor_bindings.clear();
        for spec in ACTOR_SPECS.iter().copied() {
            let id = agent
                .new_actor::<2, 2>(DeviceType::Cpu, max_traj_length, None)
                .await?;
            self.actor_bindings.push(ActorBinding {
                id,
                spec,
                last_reward: 0.0,
            });
        }
        Ok(())
    }

    pub async fn run_step_driven_episode(
        &mut self,
        agent: &RelayRLAgent<NdArray>,
        steps: usize,
    ) -> Result<(), Box<dyn Error>> {
        self.bridge.reset()?;
        for _ in 0..steps {
            let before = self.bridge.snapshot()?;
            let mut pending = Vec::with_capacity(self.actor_bindings.len());

            for binding in &self.actor_bindings {
                let observation = project_observation(&before, &binding.spec);
                let tensor = observation_tensor(observation);
                let action = agent
                    .request_action::<2, 2, Float, Float>(
                        vec![binding.id],
                        tensor,
                        None,
                        binding.last_reward,
                    )
                    .await?
                    .into_iter()
                    .next()
                    .map(|(_, action)| action);

                let (action_index, confidence) = action
                    .as_ref()
                    .and_then(|action| action.get_act())
                    .map(|tensor_data| {
                        (
                            decode_action_index(&tensor_data.data, ACTION_DIM),
                            decode_action_confidence(
                                &tensor_data.data,
                                decode_action_index(&tensor_data.data, ACTION_DIM),
                            ),
                        )
                    })
                    .unwrap_or((0, 1.0));
                pending.push((binding.id, binding.spec, action_index, confidence));
            }

            for (_, spec, action_index, confidence) in &pending {
                let command = command_from_action(spec, *action_index, *confidence);
                self.bridge.apply_command(&command)?;
            }

            let after = self.bridge.step()?;
            self.last_signals.clear();
            for binding in &mut self.actor_bindings {
                let reward = reward_for_transition(&before, &after, &binding.spec);
                binding.last_reward = reward;
                let (_, spec, action_index, _) = pending
                    .iter()
                    .find(|(id, _, _, _)| *id == binding.id)
                    .expect("pending action should exist for binding");
                let command = command_from_action(spec, *action_index, 1.0);
                self.last_signals.push(CrossRuntimeSignal {
                    source_actor: binding.spec.actor_name,
                    subsystem: binding.spec.subsystem,
                    action_index: *action_index,
                    command_description: command.description,
                    reward_after_world_step: reward,
                });
            }

            if after.done(&self.config) {
                let ids = self
                    .actor_bindings
                    .iter()
                    .map(|binding| binding.id)
                    .collect();
                agent
                    .flag_last_action(ids, Some(after.profit_delta / 1_000.0))
                    .await?;
                break;
            }
        }
        Ok(())
    }
}

pub fn objective_summary() -> BTreeMap<GameSubsystem, Vec<&'static str>> {
    let mut map = BTreeMap::new();
    for spec in ACTOR_SPECS {
        map.entry(spec.subsystem)
            .or_insert_with(Vec::new)
            .push(spec.optimizes);
    }
    map
}

pub fn print_system_map() {
    println!("RelayRL/OpenTTD actor graph:");
    for subsystem in GameSubsystem::ALL {
        println!("  {subsystem}:");
        for spec in ACTOR_SPECS
            .iter()
            .filter(|spec| spec.subsystem == subsystem)
        {
            println!(
                "    - {} optimizes {}; {}",
                spec.actor_name, spec.optimizes, spec.responsibility
            );
        }
    }
}

fn observation_tensor(observation: [f32; OBSERVATION_DIM]) -> Tensor<NdArray, 2, Float> {
    Tensor::<NdArray, 2, Float>::from_data(
        BurnTensorData::new(observation.to_vec(), [1, OBSERVATION_DIM]),
        &Default::default(),
    )
}

pub fn observation_bytes_for_actor(spec: &SubsystemActorSpec) -> Vec<u8> {
    f32_slice_to_bytes(&project_observation(
        &crate::openttd::OpenTtdSnapshot::default(),
        spec,
    ))
}
