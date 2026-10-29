## Cyber AI Agents

* CrewAI - AI agent with VM as a tool
* Swarms
* Multi-Agent Reinforcement Learning (MARL) — Agents learning to cooperate or compete through reward functions
* 

## Swarm HW


# Homework: AI Agent Swarm for Cybersecurity

## Objective

Build a simple **multi-agent AI swarm** using Ollama (Llama), Docker, and an Ubuntu Virtual Machine.

Agents must autonomously communicate, coordinate, create subgoals, and work toward a common objective using a **shared blackboard**.

**Teams:** 2 students per group.

## 1. Architecture

- **Attackers:** Two or more AI agents operating Linux Docker containers.
- **Defender:** One AI agent operating an Ubuntu VirtualBox VM.
- **LLM:** Ollama running Llama locally.
- **Communication:** Shared blackboard (Python dictionary, JSON, or SQLite).
- **Tools:** SSH (Paramiko) for interacting with Docker containers and the VM.
- **Network:** Isolated lab environment.

## 2. Shared Blackboard

The blackboard stores information that agents use to coordinate.

```python
board = {
    "goals": [],
    "subgoals": [],
    "messages": [],
    "goal_completed": False
}
```

Agents should be able to:

1. Read the blackboard.
2. Check goals and existing subgoals.
3. Communicate with other agents.
4. Create, assign, and update subgoals.
5. Record actions and observations.
6. Determine when goals have been achieved.

Attacker agents should share a team blackboard. The defender should maintain a separate blackboard.

## 3. Autonomous Agent Loop

Each agent continuously performs the following steps:

```python
while not goal_completed:

    # Read blackboard
    # Check goals and subgoals
    # Ask Llama to decide next action
    # Execute action using SSH
    # Update blackboard
    # Communicate with other agents
    # Check stopping criteria
```

Agents must make decisions using the LLM rather than following a predefined sequence of actions.

**No reinforcement learning or model training is required.**

## 4. Cybersecurity Goals

**Attacker Swarm**

- Discover services on the defender VM.
- Coordinate reconnaissance tasks.
- Find a designated test flag in the authorized lab environment.

**Defender Agent**

- Monitor system activity and logs.
- Detect suspicious behavior.
- Respond to attacker actions.
- Protect the target service.

Students may define additional goals and subgoals.

## 5. Stopping Criteria

The swarm must stop when:

- The main goal is achieved and verified.
- The maximum number of iterations is reached.
- A predefined time limit is exceeded.

## 6. Deliverables

1. Python source code for the agents and blackboard.
2. Working demonstration of the multi-agent swarm.
3. Evidence of communication and subgoal coordination.
4. Logs showing agent decisions and actions.
5. Short report describing the architecture, results, and observations.

## 7. Expected Learning Outcomes

Students will gain experience with:

- LLM-based autonomous agents
- Multi-agent systems and swarm coordination
- Blackboard communication architectures
- Goal decomposition and task allocation
- Tool execution through SSH
- AI applications in cybersecurity

**Important:** The objective is not to develop sophisticated attacks. The primary goal is to demonstrate how multiple autonomous AI agents can reason, communicate, coordinate, and pursue shared objectives in a controlled cybersecurity environment.


```



import ollama
import time

# Shared blackboard
board = {
    "goals": ["Find the test flag"],
    "subgoals": [],
    "messages": [],
    "goal_completed": False
}

def agent(name):
    response = ollama.chat(
        model="llama3.2",
        messages=[{
            "role": "user",
            "content": f"""
            You are {name}.

            Blackboard: {board}

            1. Check goals and subgoals.
            2. Read messages from other agents.
            3. Decide your next action.
            4. Create subgoals if needed.
            5. Communicate with other agents.
            6. Determine if the main goal is completed.

            Respond with your decision.
            """
        }]
    )

    decision = response["message"]["content"]

    board["messages"].append({
        "agent": name,
        "decision": decision
    })

    print(name, decision)


# Swarm loop with stopping criteria
max_rounds = 20

for round_num in range(max_rounds):

    agent("Attacker1")
    agent("Attacker2")
    agent("Defender")

    # Stop when goal is completed
    if board["goal_completed"]:
        print("Goal completed!")
        break

    time.sleep(5)

print("Swarm finished.")


```
