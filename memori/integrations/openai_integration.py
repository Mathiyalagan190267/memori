"""
OpenAI Integration - Automatic Interception System with Multi-Tenancy Support

This module provides automatic interception of OpenAI API calls when Memori is enabled.
Uses contextvars for proper isolation between concurrent requests in multi-tenant environments.

Usage:
    from openai import OpenAI
    from memori import Memori
    from memori.integrations.openai_integration import set_active_memori_context

    # Initialize Memori and enable it
    openai_memory = Memori(
        database_connect="sqlite:///openai_memory.db",
        conscious_ingest=True,
        verbose=True,
    )
    openai_memory.enable()

    # Set as active for this request context (important for multi-tenancy!)
    set_active_memori_context(openai_memory)

    # Use standard OpenAI client - automatically intercepted!
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    # Conversation is automatically recorded to the active Memori instance only
"""

import contextvars
from loguru import logger

# Context variable for tracking the active Memori instance per request
# This ensures proper isolation in multi-tenant environments
_active_memori_context = contextvars.ContextVar('active_memori', default=None)

# Global registry of enabled Memori instances (kept for backward compatibility)
_enabled_memori_instances = []


class OpenAIInterceptor:
    """
    Automatic OpenAI interception system that patches the OpenAI module
    to automatically record conversations when Memori instances are enabled.
    """

    _original_methods = {}
    _is_patched = False

    @classmethod
    def patch_openai(cls):
        """Patch OpenAI module to intercept API calls."""
        if cls._is_patched:
            return

        try:
            import openai

            # Patch sync OpenAI client
            if hasattr(openai, "OpenAI"):
                cls._patch_client_class(openai.OpenAI, "sync")

            # Patch async OpenAI client
            if hasattr(openai, "AsyncOpenAI"):
                cls._patch_async_client_class(openai.AsyncOpenAI, "async")

            # Patch Azure clients if available
            if hasattr(openai, "AzureOpenAI"):
                cls._patch_client_class(openai.AzureOpenAI, "azure_sync")

            if hasattr(openai, "AsyncAzureOpenAI"):
                cls._patch_async_client_class(openai.AsyncAzureOpenAI, "azure_async")

            cls._is_patched = True
            logger.debug("OpenAI module patched for automatic interception")

        except ImportError:
            logger.warning("OpenAI not available - skipping patch")
        except Exception as e:
            logger.error(f"Failed to patch OpenAI module: {e}")

    @classmethod
    def _patch_client_class(cls, client_class, client_type):
        """Patch a sync OpenAI client class."""
        # Store the original unbound method
        original_key = f"{client_type}_process_response"
        if original_key not in cls._original_methods:
            cls._original_methods[original_key] = client_class._process_response

        original_prepare_key = f"{client_type}_prepare_options"
        if original_prepare_key not in cls._original_methods and hasattr(
            client_class, "_prepare_options"
        ):
            cls._original_methods[original_prepare_key] = client_class._prepare_options

        # Get reference to original method to avoid recursion
        original_process = cls._original_methods[original_key]

        def patched_process_response(
            self, *, cast_to, options, response, stream, stream_cls, **kwargs
        ):
            # Call original method first with all kwargs
            result = original_process(
                self,
                cast_to=cast_to,
                options=options,
                response=response,
                stream=stream,
                stream_cls=stream_cls,
                **kwargs,
            )

            # Record conversation for enabled Memori instances
            if not stream:  # Don't record streaming here - handle separately
                cls._record_conversation_for_enabled_instances(
                    options, result, client_type
                )

            return result

        client_class._process_response = patched_process_response

        # Patch prepare_options if it exists
        if original_prepare_key in cls._original_methods:
            original_prepare = cls._original_methods[original_prepare_key]

            def patched_prepare_options(self, options):
                # Call original method first
                options = original_prepare(self, options)

                # Inject context for enabled Memori instances
                options = cls._inject_context_for_enabled_instances(
                    options, client_type
                )

                return options

            client_class._prepare_options = patched_prepare_options

    @classmethod
    def _patch_async_client_class(cls, client_class, client_type):
        """Patch an async OpenAI client class."""
        # Store the original unbound method
        original_key = f"{client_type}_process_response"
        if original_key not in cls._original_methods:
            cls._original_methods[original_key] = client_class._process_response

        original_prepare_key = f"{client_type}_prepare_options"
        if original_prepare_key not in cls._original_methods and hasattr(
            client_class, "_prepare_options"
        ):
            cls._original_methods[original_prepare_key] = client_class._prepare_options

        # Get reference to original method to avoid recursion
        original_process = cls._original_methods[original_key]

        async def patched_async_process_response(
            self, *, cast_to, options, response, stream, stream_cls, **kwargs
        ):
            # Call original method first with all kwargs
            result = await original_process(
                self,
                cast_to=cast_to,
                options=options,
                response=response,
                stream=stream,
                stream_cls=stream_cls,
                **kwargs,
            )

            # Record conversation for enabled Memori instances
            if not stream:
                cls._record_conversation_for_enabled_instances(
                    options, result, client_type
                )

            return result

        client_class._process_response = patched_async_process_response

        # Patch prepare_options if it exists
        if original_prepare_key in cls._original_methods:
            original_prepare = cls._original_methods[original_prepare_key]

            def patched_async_prepare_options(self, options):
                # Call original method first
                options = original_prepare(self, options)

                # Inject context for enabled Memori instances
                options = cls._inject_context_for_enabled_instances(
                    options, client_type
                )

                return options

            client_class._prepare_options = patched_async_prepare_options

    @classmethod
    def _inject_context_for_enabled_instances(cls, options, client_type):
        """Inject context for the active Memori instance in current request context."""
        # Use ContextVar to get the active instance for this request (multi-tenant safe)
        memori_instance = _active_memori_context.get()

        # BACKWARD COMPATIBILITY: Fallback to single global instance if context not set
        if memori_instance is None and len(_enabled_memori_instances) == 1:
            memori_instance = _enabled_memori_instances[0]
            logger.debug("Using backward-compatible single-instance mode for context injection")

        if memori_instance and memori_instance.is_enabled and (
            memori_instance.conscious_ingest or memori_instance.auto_ingest
        ):
            try:
                # Get json_data from options - handle multiple attribute name possibilities
                json_data = None
                for attr_name in ["json_data", "_json_data", "data"]:
                    if hasattr(options, attr_name):
                        json_data = getattr(options, attr_name, None)
                        if json_data:
                            break

                if not json_data:
                    # Try to reconstruct from other options attributes
                    json_data = {}
                    if hasattr(options, "messages"):
                        json_data["messages"] = options.messages
                    elif hasattr(options, "_messages"):
                        json_data["messages"] = options._messages

                if json_data and "messages" in json_data:
                    # This is a chat completion request - inject context
                    logger.debug(
                        f"OpenAI: Injecting context for {client_type} with {len(json_data['messages'])} messages"
                    )
                    updated_data = memori_instance._inject_openai_context(
                        {"messages": json_data["messages"]}
                    )

                    if updated_data.get("messages"):
                        # Update the options with modified messages
                        if hasattr(options, "json_data") and options.json_data:
                            options.json_data["messages"] = updated_data["messages"]
                        elif hasattr(options, "messages"):
                            options.messages = updated_data["messages"]

                        logger.debug(
                            f"OpenAI: Successfully injected context for {client_type}"
                        )
                else:
                    logger.debug(
                        f"OpenAI: No messages found in options for {client_type}, skipping context injection"
                    )

            except Exception as e:
                logger.error(f"Context injection failed for {client_type}: {e}")

        return options

    @classmethod
    def _is_internal_agent_call(cls, json_data):
        """
        Check if this is an internal agent processing call using multiple signals.

        Uses a combination of indicators to prevent false positives:
        1. Internal processing patterns in content
        2. Agent-specific message structure (system + user with processing instructions)
        3. JSON response format requests (agents request structured JSON output)

        This prevents legitimate user messages that mention memory processing
        from being incorrectly filtered.
        """
        try:
            messages = json_data.get("messages", [])
            if not messages:
                return False

            # Signal 1: Count pattern matches
            pattern_matches = 0
            internal_patterns = [
                "Process this conversation for enhanced memory storage:",
                "INTERNAL_MEMORY_PROCESSING:",
                "AGENT_PROCESSING_MODE:",
                "[INTERNAL_MEMORI_SEARCH]",  # Retrieval agent marker for memory searches
            ]

            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    for pattern in internal_patterns:
                        if pattern in content:
                            pattern_matches += 1

            # Signal 2: Check for agent-specific message structure
            has_agent_structure = False
            if len(messages) >= 2:
                # Agents typically have system message mentioning "memory" + "agent"
                system_msg = messages[0] if messages[0].get("role") == "system" else None
                if system_msg:
                    system_content = system_msg.get("content", "").lower()
                    if "memory" in system_content and ("agent" in system_content or "processing" in system_content):
                        has_agent_structure = True

            # Signal 3: Check for JSON response format requests (agents always request JSON)
            has_json_request = False
            for message in messages:
                if message.get("role") == "system":
                    content = message.get("content", "").lower()
                    if "json" in content and ("respond" in content or "format" in content or "output" in content):
                        has_json_request = True
                        break

            # Require pattern match AND at least one other confirming signal
            # This prevents false positives while maintaining accurate detection
            if pattern_matches >= 1 and (has_agent_structure or has_json_request):
                logger.debug(
                    f"Internal agent call detected: patterns={pattern_matches}, "
                    f"agent_structure={has_agent_structure}, json_request={has_json_request}"
                )
                return True

            # If only pattern matches without confirmation, log and treat as user message
            if pattern_matches >= 1:
                logger.debug(
                    f"Pattern found but no confirmation signals - treating as user message"
                )

            return False

        except Exception as e:
            logger.debug(f"Failed to check internal agent call: {e}")
            return False  # Fail open - record if uncertain to avoid data loss

    @classmethod
    def _record_conversation_for_enabled_instances(cls, options, response, client_type):
        """
        Record conversation to the active Memori instance in the current context.

        This ensures proper isolation in multi-tenant environments by only recording
        to the instance that's active for the current request, not all enabled instances.
        """
        # Get the active Memori instance for THIS request context
        active_memori = _active_memori_context.get()

        if active_memori and active_memori.is_enabled:
            try:
                json_data = getattr(options, "json_data", None) or {}

                if "messages" in json_data:
                    # Check if this is an internal agent processing call
                    is_internal = cls._is_internal_agent_call(json_data)

                    # Debug logging to help diagnose recording issues
                    user_messages = [
                        msg
                        for msg in json_data.get("messages", [])
                        if msg.get("role") == "user"
                    ]
                    if user_messages and not is_internal:
                        user_content = user_messages[-1].get("content", "")[:50]
                        logger.debug(
                            f"Recording to active instance: '{user_content}...' (user_id={getattr(active_memori, 'user_id', 'unknown')})"
                        )
                    elif is_internal:
                        logger.debug(
                            "Skipping internal agent call (detected pattern match)"
                        )

                    # Skip internal agent processing calls
                    if is_internal:
                        return

                    # Record to the active instance only
                    active_memori._record_openai_conversation(json_data, response)
                    logger.debug(f"Conversation recorded to instance: {getattr(active_memori, 'user_id', 'unknown')}")

                elif "prompt" in json_data:
                    # Legacy completions
                    cls._record_legacy_completion(
                        active_memori, json_data, response, client_type
                    )

            except Exception as e:
                logger.error(
                    f"Failed to record conversation for {client_type}: {e}"
                )
        # BACKWARD COMPATIBILITY: Fallback to global instance for single-user apps
        elif len(_enabled_memori_instances) == 1:
            single_instance = _enabled_memori_instances[0]
            if single_instance.is_enabled:
                try:
                    logger.debug(
                        "Using backward-compatible single-instance mode (context not set). "
                        "For multi-tenant apps, call set_active_memori_context(memori) explicitly."
                    )
                    json_data = getattr(options, "json_data", None) or {}

                    if "messages" in json_data:
                        is_internal = cls._is_internal_agent_call(json_data)
                        if not is_internal:
                            single_instance._record_openai_conversation(json_data, response)
                            logger.debug("Conversation recorded to single global instance (backward compatibility mode)")

                    elif "prompt" in json_data:
                        cls._record_legacy_completion(single_instance, json_data, response, client_type)

                except Exception as e:
                    logger.error(f"Failed to record conversation (legacy mode): {e}")

        # Multiple instances without context: Raise error (correct behavior)
        elif len(_enabled_memori_instances) > 1:
            logger.error(
                "❌ MULTI-TENANT ISOLATION ERROR: Multiple Memori instances registered but no active context set. "
                "This is ambiguous and prevents proper tenant isolation. "
                "Call set_active_memori_context(memori) before making OpenAI API calls in multi-tenant environments.\n\n"
                "Example:\n"
                "  from memori.integrations.openai_integration import set_active_memori_context\n"
                "  memori = Memori(user_id='user123', ...)\n"
                "  memori.enable()\n"
                "  set_active_memori_context(memori)  # Required for multi-tenant!\n"
                "  # Now make your OpenAI calls\n\n"
                "See https://docs.memori.ai/multi-tenant for more info."
            )

        # No instances enabled: Silent skip (normal)
        else:
            logger.debug("No Memori instances enabled, skipping recording")

    @classmethod
    def _record_legacy_completion(
        cls, memori_instance, request_data, response, client_type
    ):
        """Record legacy completion API calls."""
        try:
            prompt = request_data.get("prompt", "")
            model = request_data.get("model", "unknown")

            # Extract AI response
            ai_output = ""
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "text"):
                    ai_output = choice.text or ""

            # Calculate tokens
            tokens_used = 0
            if hasattr(response, "usage") and response.usage:
                tokens_used = getattr(response.usage, "total_tokens", 0)

            # Record conversation
            memori_instance.record_conversation(
                user_input=prompt,
                ai_output=ai_output,
                model=model,
                metadata={
                    "integration": "openai_auto_intercept",
                    "client_type": client_type,
                    "api_type": "completions",
                    "tokens_used": tokens_used,
                    "auto_recorded": True,
                },
            )
        except Exception as e:
            logger.error(f"Failed to record legacy completion: {e}")

    @classmethod
    def unpatch_openai(cls):
        """Restore original OpenAI module methods."""
        if not cls._is_patched:
            return

        try:
            import openai

            # Restore sync OpenAI client
            if "sync_process_response" in cls._original_methods:
                openai.OpenAI._process_response = cls._original_methods[
                    "sync_process_response"
                ]

            if "sync_prepare_options" in cls._original_methods:
                openai.OpenAI._prepare_options = cls._original_methods[
                    "sync_prepare_options"
                ]

            # Restore async OpenAI client
            if "async_process_response" in cls._original_methods:
                openai.AsyncOpenAI._process_response = cls._original_methods[
                    "async_process_response"
                ]

            if "async_prepare_options" in cls._original_methods:
                openai.AsyncOpenAI._prepare_options = cls._original_methods[
                    "async_prepare_options"
                ]

            # Restore Azure clients
            if (
                hasattr(openai, "AzureOpenAI")
                and "azure_sync_process_response" in cls._original_methods
            ):
                openai.AzureOpenAI._process_response = cls._original_methods[
                    "azure_sync_process_response"
                ]

            if (
                hasattr(openai, "AzureOpenAI")
                and "azure_sync_prepare_options" in cls._original_methods
            ):
                openai.AzureOpenAI._prepare_options = cls._original_methods[
                    "azure_sync_prepare_options"
                ]

            if (
                hasattr(openai, "AsyncAzureOpenAI")
                and "azure_async_process_response" in cls._original_methods
            ):
                openai.AsyncAzureOpenAI._process_response = cls._original_methods[
                    "azure_async_process_response"
                ]

            if (
                hasattr(openai, "AsyncAzureOpenAI")
                and "azure_async_prepare_options" in cls._original_methods
            ):
                openai.AsyncAzureOpenAI._prepare_options = cls._original_methods[
                    "azure_async_prepare_options"
                ]

            cls._is_patched = False
            cls._original_methods.clear()
            logger.debug("OpenAI module patches removed")

        except ImportError:
            pass  # OpenAI not available
        except Exception as e:
            logger.error(f"Failed to unpatch OpenAI module: {e}")


def register_memori_instance(memori_instance):
    """
    DEPRECATED: Register a Memori instance for automatic OpenAI interception.

    This function is deprecated and will be removed in v2.0.0.
    Use set_active_memori_context() instead for proper multi-tenant isolation.

    Args:
        memori_instance: Memori instance to register
    """
    import warnings
    warnings.warn(
        "register_memori_instance() is deprecated and will be removed in v2.0.0. "
        "Use set_active_memori_context(memori) instead for proper multi-tenant isolation. "
        "See https://docs.memori.ai/multi-tenant for migration guide.",
        DeprecationWarning,
        stacklevel=2
    )

    global _enabled_memori_instances

    if memori_instance not in _enabled_memori_instances:
        _enabled_memori_instances.append(memori_instance)
        logger.debug("⚠️  Registered Memori instance using deprecated method")

    # Ensure OpenAI is patched
    OpenAIInterceptor.patch_openai()


def unregister_memori_instance(memori_instance):
    """
    DEPRECATED: Unregister a Memori instance from automatic OpenAI interception.

    This function is deprecated and will be removed in v2.0.0.
    Use clear_active_memori_context() instead for proper multi-tenant isolation.

    Args:
        memori_instance: Memori instance to unregister
    """
    import warnings
    warnings.warn(
        "unregister_memori_instance() is deprecated and will be removed in v2.0.0. "
        "Use clear_active_memori_context() instead (usually not needed as context auto-clears). "
        "See https://docs.memori.ai/multi-tenant for migration guide.",
        DeprecationWarning,
        stacklevel=2
    )

    global _enabled_memori_instances

    if memori_instance in _enabled_memori_instances:
        _enabled_memori_instances.remove(memori_instance)
        logger.debug("⚠️  Unregistered Memori instance using deprecated method")

    # If no more instances, unpatch OpenAI
    if not _enabled_memori_instances:
        OpenAIInterceptor.unpatch_openai()


def get_enabled_instances():
    """Get list of currently enabled Memori instances."""
    return _enabled_memori_instances.copy()


def is_openai_patched():
    """Check if OpenAI module is currently patched."""
    return OpenAIInterceptor._is_patched


# For backward compatibility - keep old classes but mark as deprecated
class MemoriOpenAI:
    """
    DEPRECATED: Legacy wrapper class.

    Use automatic interception instead:
        memori = Memori(...)
        memori.enable()
        client = OpenAI()  # Automatically intercepted
    """

    def __init__(self, memori_instance, **kwargs):
        logger.warning(
            "MemoriOpenAI is deprecated. Use automatic interception instead:\n"
            "memori.enable() then use OpenAI() client directly."
        )

        try:
            import openai

            self._openai = openai.OpenAI(**kwargs)

            # Register for automatic interception
            register_memori_instance(memori_instance)

            # Pass through all attributes
            for attr in dir(self._openai):
                if not attr.startswith("_"):
                    setattr(self, attr, getattr(self._openai, attr))

        except ImportError as err:
            raise ImportError("OpenAI package required: pip install openai") from err


class MemoriOpenAIInterceptor(MemoriOpenAI):
    """DEPRECATED: Use automatic interception instead."""

    def __init__(self, memori_instance, **kwargs):
        logger.warning(
            "MemoriOpenAIInterceptor is deprecated. Use automatic interception instead:\n"
            "memori.enable() then use OpenAI() client directly."
        )
        super().__init__(memori_instance, **kwargs)


def create_openai_client(memori_instance, provider_config=None, **kwargs):
    """
    Create an OpenAI client that automatically records to memori.

    This is the recommended way to create OpenAI clients with memori integration.

    Args:
        memori_instance: Memori instance to record conversations to
        provider_config: Provider configuration (optional)
        **kwargs: Additional arguments for OpenAI client

    Returns:
        OpenAI client instance with automatic recording
    """
    try:
        import openai

        # Register the memori instance for automatic interception
        register_memori_instance(memori_instance)

        # Use provider config if available, otherwise use kwargs
        if provider_config:
            client_kwargs = provider_config.to_openai_kwargs()
            client_kwargs.update(kwargs)  # Allow kwargs to override config
        else:
            client_kwargs = kwargs

        # Create standard OpenAI client - it will be automatically intercepted
        client = openai.OpenAI(**client_kwargs)

        logger.info("Created OpenAI client with automatic memori recording")
        return client

    except ImportError as e:
        logger.error(f"Failed to import OpenAI: {e}")
        raise ImportError("OpenAI package required: pip install openai") from e
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {e}")
        raise


# Multi-Tenancy Context Management Functions

def set_active_memori_context(memori_instance):
    """
    Set the active Memori instance for the current request context.

    This ensures that OpenAI API calls made within this context are recorded
    ONLY to this specific Memori instance, providing proper isolation in
    multi-tenant environments.

    Args:
        memori_instance: The Memori instance to set as active

    Example:
        # In your API endpoint:
        memori = await manager.get_instance(user_id=user_id, ...)
        set_active_memori_context(memori)  # Set for this request

        # Now all OpenAI calls record to this instance only
        response = openai_client.chat.completions.create(...)
    """
    _active_memori_context.set(memori_instance)
    logger.debug(f"Set active Memori context: user_id={getattr(memori_instance, 'user_id', 'unknown')}")


def get_active_memori_context():
    """
    Get the currently active Memori instance for this request context.

    Returns:
        The active Memori instance, or None if no instance is set
    """
    return _active_memori_context.get()


def clear_active_memori_context():
    """
    Clear the active Memori instance from the current request context.

    This is usually not necessary as context is automatically cleared when
    the request completes, but can be called explicitly if needed.
    """
    _active_memori_context.set(None)
    logger.debug("Cleared active Memori context")


def get_enabled_instances():
    """
    Get the list of currently enabled Memori instances.

    Returns:
        list: List of enabled Memori instances (global registry)

    Note: This returns the global registry. For multi-tenant apps,
    use get_active_memori_context() to get the request-specific instance.
    """
    return _enabled_memori_instances.copy()
