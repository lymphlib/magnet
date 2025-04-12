{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}        
   :show-inheritance:

   
   {% block methods %}
   .. rubric:: {{ _('Constructor') }}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   
   {% block method_details %}
   {% if methods %}
   {% for item in methods %}
   {%- if item not in inherited_members %}
   .. automethod:: {{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {%- endif %}
   {% endblock %}

   {%- set ns = namespace(exclusively_inherited = false,
   to_exlude = [
   'add_module',
   'apply',
   'bfloat16',
   'buffers',
   'children',
   'compile',
   'cpu',
   'cuda',
   'double',
   'eval',
   'extra_repr',
   'float',
   'forward',
   'get_buffer',
   'get_extra_state',
   'get_parameter',
   'get_submodule',
   'half',
   'ipu',
   'load_state_dict',
   'modules',
   'named_buffers',
   'named_children',
   'named_modules',
   'named_parameters',
   'parameters',
   'register_backward_hook',
   'register_buffer',
   'register_forward_hook',
   'register_forward_pre_hook',
   'register_full_backward_hook',
   'register_full_backward_pre_hook',
   'register_load_state_dict_post_hook',
   'register_module',
   'register_parameter',
   'register_state_dict_pre_hook',
   'requires_grad_',
   'set_extra_state',
   'share_memory',
   'state_dict',
   'to',
   'to_empty',
   'train',
   'type',
   'xpu',
   'zero_grad']) %}
   
   {% for item in methods %}
   {%- if item in inherited_members %}
   {%- if item not in ns.to_exlude %}
   {%- set ns.exclusively_inherited = true %}
   {%- endif %}
   {%- endif %}
   {%- endfor %}


   {% block inherited_members %}
   {% if ns.exclusively_inherited %}
   .. rubric:: {{ _('Inherited Methods') }}

   .. autosummary::
   {% for item in methods %}
   {%- if item in inherited_members %}
   {%- if item not in ns.to_exlude %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   
   {% if (objname.startswith('GNN') or objname.startswith('Sage')) %}
   .. note::
      This class inherits from ``torch.nn.Module``. To see the full list of inherited members,
      please see the `Pytorch documentation <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_.
   {%- endif %}

   {% endblock %}

