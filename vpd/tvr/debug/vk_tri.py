"""V2 vk_tri: one triangle through ctypes Vulkan 1.1, offscreen, read back with vkCmdCopyImageToBuffer.

JOB probe (NVIDIA ICD) and login-node smoke (lavapipe ICD only: 1 core, LP_NUM_THREADS=1, nice -n 19).
  1. CDLL(--loader) (an absolute path picks the loader: image 1.2.131 or env 1.4.357), vkEnumerateInstanceVersion,
     vkCreateInstance (API 1.1, no layers, no extensions), every physical device with its properties;
     the device is the first one matching --want (nvidia = vendorID 0x10de, cpu = deviceType CPU, any) that
     has a graphics queue.
  2. R8G8B8A8_UNORM 256x256 colour image (OPTIMAL tiling, COLOR_ATTACHMENT | TRANSFER_SRC), image view, render
     pass with one attachment (loadOp CLEAR to (0.2, 0.4, 0.6, 1), storeOp STORE, final layout
     COLOR_ATTACHMENT_OPTIMAL), framebuffer, shader modules from shaders/tri.{vert,frag}.spv (SPIR-V 1.0,
     assembled and validated on the login node), pipeline with no vertex input, cull NONE.
  3. One command buffer: begin render pass, bind, vkCmdDraw(3), end, image barrier to TRANSFER_SRC_OPTIMAL,
     vkCmdCopyImageToBuffer into a HOST_VISIBLE | HOST_COHERENT buffer, buffer barrier to HOST_READ; one
     vkQueueSubmit with a fence, vkWaitForFences (10 s); map and read.
The VkResult of every call is recorded (metrics.vk), in particular vkCreateGraphicsPipelines, vkQueueSubmit
and vkWaitForFences.  PASS = those three VK_SUCCESS and the pixel rule of common.pixel_verdict (centre within +-3
of (255, 0, 0), all four corners within +-2 of (51, 102, 153)).  Writes OUT/result.json (+ tri.png).
No cleanup calls: the result is written first and the process exit releases everything.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import struct
import time

from . import common

u32, i32, u64, f32 = C.c_uint32, C.c_int32, C.c_uint64, C.c_float
vp, cp = C.c_void_p, C.c_char_p
H = C.c_uint64  # non-dispatchable handle
D = C.c_void_p  # dispatchable handle
P = C.POINTER

API_1_1 = (1 << 22) | (1 << 12)
SHADERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
SPV_MAGIC = 0x07230203
FENCE_TIMEOUT_NS = 10 * 1000 ** 3

# enums / flags
FORMAT_R8G8B8A8_UNORM = 37
IMAGE_TYPE_2D, IMAGE_VIEW_TYPE_2D, SAMPLE_1, TILING_OPTIMAL = 1, 1, 1, 0
USAGE_TRANSFER_SRC, USAGE_COLOR_ATTACHMENT, BUF_USAGE_TRANSFER_DST = 0x1, 0x10, 0x2
LAYOUT_UNDEFINED, LAYOUT_COLOR_ATTACHMENT, LAYOUT_TRANSFER_SRC = 0, 2, 6
MEM_DEVICE_LOCAL, MEM_HOST_VISIBLE, MEM_HOST_COHERENT = 0x1, 0x2, 0x4
ASPECT_COLOR = 0x1
LOAD_CLEAR, LOAD_DONT_CARE, STORE_STORE, STORE_DONT_CARE = 1, 2, 0, 1
BIND_GRAPHICS = 0
STAGE_VERTEX, STAGE_FRAGMENT = 0x1, 0x10
TOPOLOGY_TRIANGLE_LIST, POLYGON_FILL, CULL_NONE, FRONT_CCW = 3, 0, 0, 0
COLOR_RGBA = 0xF
QUEUE_GRAPHICS = 0x1
QUEUE_FAMILY_IGNORED = 0xFFFFFFFF
PS_COLOR_OUTPUT, PS_TRANSFER, PS_HOST = 0x400, 0x1000, 0x4000
ACC_COLOR_WRITE, ACC_TRANSFER_READ, ACC_TRANSFER_WRITE, ACC_HOST_READ = 0x100, 0x800, 0x1000, 0x2000
CB_ONE_TIME = 0x1
DEVICE_TYPES = {0: "OTHER", 1: "INTEGRATED_GPU", 2: "DISCRETE_GPU", 3: "VIRTUAL_GPU", 4: "CPU"}
VK_RESULTS = {0: "VK_SUCCESS", 1: "VK_NOT_READY", 2: "VK_TIMEOUT", 3: "VK_EVENT_SET", 4: "VK_EVENT_RESET",
              5: "VK_INCOMPLETE", -1: "VK_ERROR_OUT_OF_HOST_MEMORY", -2: "VK_ERROR_OUT_OF_DEVICE_MEMORY",
              -3: "VK_ERROR_INITIALIZATION_FAILED", -4: "VK_ERROR_DEVICE_LOST", -5: "VK_ERROR_MEMORY_MAP_FAILED",
              -6: "VK_ERROR_LAYER_NOT_PRESENT", -7: "VK_ERROR_EXTENSION_NOT_PRESENT",
              -8: "VK_ERROR_FEATURE_NOT_PRESENT", -9: "VK_ERROR_INCOMPATIBLE_DRIVER",
              -10: "VK_ERROR_TOO_MANY_OBJECTS", -11: "VK_ERROR_FORMAT_NOT_SUPPORTED", -13: "VK_ERROR_UNKNOWN",
              -1000012000: "VK_ERROR_INVALID_SHADER_NV"}
# sType values (Vulkan 1.0 core)
(ST_APP, ST_INSTANCE, ST_DEVICE_QUEUE, ST_DEVICE, ST_SUBMIT, ST_MEM_ALLOC) = (0, 1, 2, 3, 4, 5)
ST_FENCE, ST_BUFFER, ST_IMAGE, ST_IMAGE_VIEW, ST_SHADER_MODULE = 8, 12, 14, 15, 16
(ST_STAGE, ST_VERTEX_INPUT, ST_INPUT_ASSEMBLY, ST_VIEWPORT, ST_RASTER, ST_MULTISAMPLE,
 ST_BLEND, ST_GRAPHICS_PIPELINE, ST_PIPELINE_LAYOUT) = (18, 19, 20, 22, 23, 24, 26, 28, 30)
ST_FRAMEBUFFER, ST_RENDER_PASS, ST_CMD_POOL, ST_CMD_ALLOC, ST_CMD_BEGIN, ST_RP_BEGIN = 37, 38, 39, 40, 42, 43
ST_BUFFER_BARRIER, ST_IMAGE_BARRIER = 44, 45


def vk_name(r) -> str:
    return f"{VK_RESULTS.get(int(r), 'VkResult')}({int(r)})"


def version_str(v: int) -> str:
    return f"{(v >> 22) & 0x7F}.{(v >> 12) & 0x3FF}.{v & 0xFFF}"


def st(name, fields):
    return type(name, (C.Structure,), {"_fields_": fields})


AppInfo = st("AppInfo", [("sType", u32), ("pNext", vp), ("pApplicationName", cp), ("applicationVersion", u32),
                         ("pEngineName", cp), ("engineVersion", u32), ("apiVersion", u32)])
InstanceCI = st("InstanceCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("pApplicationInfo", P(AppInfo)),
                               ("enabledLayerCount", u32), ("ppEnabledLayerNames", P(cp)),
                               ("enabledExtensionCount", u32), ("ppEnabledExtensionNames", P(cp))])
QueueFamilyProps = st("QueueFamilyProps", [("queueFlags", u32), ("queueCount", u32), ("timestampValidBits", u32),
                                           ("granW", u32), ("granH", u32), ("granD", u32)])
MemType = st("MemType", [("propertyFlags", u32), ("heapIndex", u32)])
MemHeap = st("MemHeap", [("size", u64), ("flags", u32)])
MemProps = st("MemProps", [("memoryTypeCount", u32), ("memoryTypes", MemType * 32), ("memoryHeapCount", u32),
                           ("memoryHeaps", MemHeap * 16)])
DeviceQueueCI = st("DeviceQueueCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("queueFamilyIndex", u32),
                                     ("queueCount", u32), ("pQueuePriorities", P(f32))])
DeviceCI = st("DeviceCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("queueCreateInfoCount", u32),
                           ("pQueueCreateInfos", P(DeviceQueueCI)), ("enabledLayerCount", u32),
                           ("ppEnabledLayerNames", P(cp)), ("enabledExtensionCount", u32),
                           ("ppEnabledExtensionNames", P(cp)), ("pEnabledFeatures", vp)])
Extent2D = st("Extent2D", [("width", u32), ("height", u32)])
Extent3D = st("Extent3D", [("width", u32), ("height", u32), ("depth", u32)])
Offset2D = st("Offset2D", [("x", i32), ("y", i32)])
Offset3D = st("Offset3D", [("x", i32), ("y", i32), ("z", i32)])
Rect2D = st("Rect2D", [("offset", Offset2D), ("extent", Extent2D)])
ImageCI = st("ImageCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("imageType", u32), ("format", u32),
                         ("extent", Extent3D), ("mipLevels", u32), ("arrayLayers", u32), ("samples", u32),
                         ("tiling", u32), ("usage", u32), ("sharingMode", u32), ("queueFamilyIndexCount", u32),
                         ("pQueueFamilyIndices", P(u32)), ("initialLayout", u32)])
MemReqs = st("MemReqs", [("size", u64), ("alignment", u64), ("memoryTypeBits", u32)])
MemAllocInfo = st("MemAllocInfo", [("sType", u32), ("pNext", vp), ("allocationSize", u64),
                                   ("memoryTypeIndex", u32)])
ComponentMapping = st("ComponentMapping", [("r", u32), ("g", u32), ("b", u32), ("a", u32)])
SubresRange = st("SubresRange", [("aspectMask", u32), ("baseMipLevel", u32), ("levelCount", u32),
                                 ("baseArrayLayer", u32), ("layerCount", u32)])
ImageViewCI = st("ImageViewCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("image", H), ("viewType", u32),
                                 ("format", u32), ("components", ComponentMapping),
                                 ("subresourceRange", SubresRange)])
AttachmentDesc = st("AttachmentDesc", [("flags", u32), ("format", u32), ("samples", u32), ("loadOp", u32),
                                       ("storeOp", u32), ("stencilLoadOp", u32), ("stencilStoreOp", u32),
                                       ("initialLayout", u32), ("finalLayout", u32)])
AttachmentRef = st("AttachmentRef", [("attachment", u32), ("layout", u32)])
SubpassDesc = st("SubpassDesc", [("flags", u32), ("pipelineBindPoint", u32), ("inputAttachmentCount", u32),
                                 ("pInputAttachments", P(AttachmentRef)), ("colorAttachmentCount", u32),
                                 ("pColorAttachments", P(AttachmentRef)), ("pResolveAttachments", P(AttachmentRef)),
                                 ("pDepthStencilAttachment", P(AttachmentRef)), ("preserveAttachmentCount", u32),
                                 ("pPreserveAttachments", P(u32))])
RenderPassCI = st("RenderPassCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("attachmentCount", u32),
                                   ("pAttachments", P(AttachmentDesc)), ("subpassCount", u32),
                                   ("pSubpasses", P(SubpassDesc)), ("dependencyCount", u32), ("pDependencies", vp)])
FramebufferCI = st("FramebufferCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("renderPass", H),
                                     ("attachmentCount", u32), ("pAttachments", P(H)), ("width", u32),
                                     ("height", u32), ("layers", u32)])
ShaderModuleCI = st("ShaderModuleCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("codeSize", C.c_size_t),
                                       ("pCode", P(u32))])
ShaderStageCI = st("ShaderStageCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("stage", u32),
                                     ("module", H), ("pName", cp), ("pSpecializationInfo", vp)])
VertexInputCI = st("VertexInputCI", [("sType", u32), ("pNext", vp), ("flags", u32),
                                     ("vertexBindingDescriptionCount", u32), ("pVertexBindingDescriptions", vp),
                                     ("vertexAttributeDescriptionCount", u32),
                                     ("pVertexAttributeDescriptions", vp)])
InputAssemblyCI = st("InputAssemblyCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("topology", u32),
                                         ("primitiveRestartEnable", u32)])
Viewport = st("Viewport", [("x", f32), ("y", f32), ("width", f32), ("height", f32), ("minDepth", f32),
                           ("maxDepth", f32)])
ViewportCI = st("ViewportCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("viewportCount", u32),
                               ("pViewports", P(Viewport)), ("scissorCount", u32), ("pScissors", P(Rect2D))])
RasterCI = st("RasterCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("depthClampEnable", u32),
                           ("rasterizerDiscardEnable", u32), ("polygonMode", u32), ("cullMode", u32),
                           ("frontFace", u32), ("depthBiasEnable", u32), ("depthBiasConstantFactor", f32),
                           ("depthBiasClamp", f32), ("depthBiasSlopeFactor", f32), ("lineWidth", f32)])
MultisampleCI = st("MultisampleCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("rasterizationSamples", u32),
                                     ("sampleShadingEnable", u32), ("minSampleShading", f32), ("pSampleMask", vp),
                                     ("alphaToCoverageEnable", u32), ("alphaToOneEnable", u32)])
BlendAttachment = st("BlendAttachment", [("blendEnable", u32), ("srcColorBlendFactor", u32),
                                         ("dstColorBlendFactor", u32), ("colorBlendOp", u32),
                                         ("srcAlphaBlendFactor", u32), ("dstAlphaBlendFactor", u32),
                                         ("alphaBlendOp", u32), ("colorWriteMask", u32)])
BlendCI = st("BlendCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("logicOpEnable", u32), ("logicOp", u32),
                         ("attachmentCount", u32), ("pAttachments", P(BlendAttachment)),
                         ("blendConstants", f32 * 4)])
PipelineLayoutCI = st("PipelineLayoutCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("setLayoutCount", u32),
                                           ("pSetLayouts", vp), ("pushConstantRangeCount", u32),
                                           ("pPushConstantRanges", vp)])
GraphicsPipelineCI = st("GraphicsPipelineCI", [
    ("sType", u32), ("pNext", vp), ("flags", u32), ("stageCount", u32), ("pStages", P(ShaderStageCI)),
    ("pVertexInputState", P(VertexInputCI)), ("pInputAssemblyState", P(InputAssemblyCI)),
    ("pTessellationState", vp), ("pViewportState", P(ViewportCI)), ("pRasterizationState", P(RasterCI)),
    ("pMultisampleState", P(MultisampleCI)), ("pDepthStencilState", vp), ("pColorBlendState", P(BlendCI)),
    ("pDynamicState", vp), ("layout", H), ("renderPass", H), ("subpass", u32), ("basePipelineHandle", H),
    ("basePipelineIndex", i32)])
CommandPoolCI = st("CommandPoolCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("queueFamilyIndex", u32)])
CommandBufferAI = st("CommandBufferAI", [("sType", u32), ("pNext", vp), ("commandPool", H), ("level", u32),
                                         ("commandBufferCount", u32)])
CommandBufferBI = st("CommandBufferBI", [("sType", u32), ("pNext", vp), ("flags", u32), ("pInheritanceInfo", vp)])
ClearValue = st("ClearValue", [("float32", f32 * 4)])  # the VkClearValue union is 16 bytes, 4-aligned
RenderPassBI = st("RenderPassBI", [("sType", u32), ("pNext", vp), ("renderPass", H), ("framebuffer", H),
                                   ("renderArea", Rect2D), ("clearValueCount", u32),
                                   ("pClearValues", P(ClearValue))])
ImageBarrier = st("ImageBarrier", [("sType", u32), ("pNext", vp), ("srcAccessMask", u32), ("dstAccessMask", u32),
                                   ("oldLayout", u32), ("newLayout", u32), ("srcQueueFamilyIndex", u32),
                                   ("dstQueueFamilyIndex", u32), ("image", H), ("subresourceRange", SubresRange)])
BufferBarrier = st("BufferBarrier", [("sType", u32), ("pNext", vp), ("srcAccessMask", u32), ("dstAccessMask", u32),
                                     ("srcQueueFamilyIndex", u32), ("dstQueueFamilyIndex", u32), ("buffer", H),
                                     ("offset", u64), ("size", u64)])
BufferCI = st("BufferCI", [("sType", u32), ("pNext", vp), ("flags", u32), ("size", u64), ("usage", u32),
                           ("sharingMode", u32), ("queueFamilyIndexCount", u32), ("pQueueFamilyIndices", P(u32))])
SubresLayers = st("SubresLayers", [("aspectMask", u32), ("mipLevel", u32), ("baseArrayLayer", u32),
                                   ("layerCount", u32)])
BufferImageCopy = st("BufferImageCopy", [("bufferOffset", u64), ("bufferRowLength", u32),
                                         ("bufferImageHeight", u32), ("imageSubresource", SubresLayers),
                                         ("imageOffset", Offset3D), ("imageExtent", Extent3D)])
SubmitInfo = st("SubmitInfo", [("sType", u32), ("pNext", vp), ("waitSemaphoreCount", u32), ("pWaitSemaphores", vp),
                               ("pWaitDstStageMask", vp), ("commandBufferCount", u32), ("pCommandBuffers", P(D)),
                               ("signalSemaphoreCount", u32), ("pSignalSemaphores", vp)])
FenceCI = st("FenceCI", [("sType", u32), ("pNext", vp), ("flags", u32)])

# C sizes on x86_64 Linux (checked by the tests against the Vulkan headers' layout)
EXPECTED_SIZES = {"AppInfo": 48, "InstanceCI": 64, "DeviceQueueCI": 40, "DeviceCI": 72, "ImageCI": 88,
                  "MemReqs": 24, "MemProps": 520, "ImageViewCI": 80, "AttachmentDesc": 36, "SubpassDesc": 72,
                  "RenderPassCI": 64, "FramebufferCI": 64, "ShaderModuleCI": 40, "ShaderStageCI": 48,
                  "VertexInputCI": 48, "InputAssemblyCI": 32, "ViewportCI": 48, "RasterCI": 64,
                  "MultisampleCI": 48, "BlendAttachment": 32, "BlendCI": 56, "PipelineLayoutCI": 48,
                  "GraphicsPipelineCI": 144, "CommandPoolCI": 24, "CommandBufferAI": 32, "CommandBufferBI": 32,
                  "ClearValue": 16, "RenderPassBI": 64, "ImageBarrier": 72, "BufferBarrier": 56, "BufferCI": 56,
                  "BufferImageCopy": 56, "SubmitInfo": 72, "FenceCI": 24, "QueueFamilyProps": 24}

PROTOS = {
    "vkEnumerateInstanceVersion": (i32, P(u32)),
    "vkCreateInstance": (i32, P(InstanceCI), vp, P(D)),
    "vkEnumeratePhysicalDevices": (i32, D, P(u32), P(D)),
    "vkGetPhysicalDeviceProperties": (None, D, vp),
    "vkGetPhysicalDeviceQueueFamilyProperties": (None, D, P(u32), P(QueueFamilyProps)),
    "vkGetPhysicalDeviceMemoryProperties": (None, D, P(MemProps)),
    "vkCreateDevice": (i32, D, P(DeviceCI), vp, P(D)),
    "vkGetDeviceQueue": (None, D, u32, u32, P(D)),
    "vkCreateImage": (i32, D, P(ImageCI), vp, P(H)),
    "vkGetImageMemoryRequirements": (None, D, H, P(MemReqs)),
    "vkAllocateMemory": (i32, D, P(MemAllocInfo), vp, P(H)),
    "vkBindImageMemory": (i32, D, H, H, u64),
    "vkCreateImageView": (i32, D, P(ImageViewCI), vp, P(H)),
    "vkCreateRenderPass": (i32, D, P(RenderPassCI), vp, P(H)),
    "vkCreateFramebuffer": (i32, D, P(FramebufferCI), vp, P(H)),
    "vkCreateShaderModule": (i32, D, P(ShaderModuleCI), vp, P(H)),
    "vkCreatePipelineLayout": (i32, D, P(PipelineLayoutCI), vp, P(H)),
    "vkCreateGraphicsPipelines": (i32, D, H, u32, P(GraphicsPipelineCI), vp, P(H)),
    "vkCreateBuffer": (i32, D, P(BufferCI), vp, P(H)),
    "vkGetBufferMemoryRequirements": (None, D, H, P(MemReqs)),
    "vkBindBufferMemory": (i32, D, H, H, u64),
    "vkCreateCommandPool": (i32, D, P(CommandPoolCI), vp, P(H)),
    "vkAllocateCommandBuffers": (i32, D, P(CommandBufferAI), P(D)),
    "vkBeginCommandBuffer": (i32, D, P(CommandBufferBI)),
    "vkCmdBeginRenderPass": (None, D, P(RenderPassBI), u32),
    "vkCmdBindPipeline": (None, D, u32, H),
    "vkCmdDraw": (None, D, u32, u32, u32, u32),
    "vkCmdEndRenderPass": (None, D),
    "vkCmdPipelineBarrier": (None, D, u32, u32, u32, u32, vp, u32, P(BufferBarrier), u32, P(ImageBarrier)),
    "vkCmdCopyImageToBuffer": (None, D, H, u32, H, u32, P(BufferImageCopy)),
    "vkEndCommandBuffer": (i32, D),
    "vkCreateFence": (i32, D, P(FenceCI), vp, P(H)),
    "vkQueueSubmit": (i32, D, u32, P(SubmitInfo), H),
    "vkWaitForFences": (i32, D, u32, P(H), u32, u64),
    "vkMapMemory": (i32, D, H, u64, u64, u32, P(vp)),
}


class _Stop(Exception):
    pass


def ptr(arr, ctype):
    """A POINTER(ctype) to the first element of a ctypes array (the cast keeps the array alive)."""
    return C.cast(arr, P(ctype))


def load_spv(path: str):
    """(uint32 array, byte length, sha256) of a SPIR-V binary; raises ValueError on a bad magic / length."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20 or len(data) % 4:
        raise ValueError(f"{path}: {len(data)} bytes is not SPIR-V")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != SPV_MAGIC:
        raise ValueError(f"{path}: magic {words[0]:#x} != {SPV_MAGIC:#x}")
    return (u32 * len(words))(*words), len(data), hashlib.sha256(data).hexdigest()


def parse_props(raw: bytes) -> dict:
    """The leading fields of VkPhysicalDeviceProperties (apiVersion .. deviceName)."""
    api, drv, vendor, device, dtype = struct.unpack_from("<5I", raw, 0)
    name = raw[20:276].split(b"\0", 1)[0].decode(errors="replace")
    return {"name": name, "type": DEVICE_TYPES.get(dtype, str(dtype)), "vendorID": f"{vendor:#06x}",
            "deviceID": f"{device:#06x}", "apiVersion": version_str(api), "driverVersion_raw": drv}


def wants(dev: dict, want: str) -> bool:
    if want == "nvidia":
        return dev["vendorID"] == "0x10de"
    if want == "cpu":
        return dev["type"] == "CPU"
    return True


def find_memory_type(props, bits: int, flags: int):
    for i in range(props.memoryTypeCount):
        if bits & (1 << i) and (props.memoryTypes[i].propertyFlags & flags) == flags:
            return i
    return None


def bind_api(lib):
    """Set restype / argtypes of the PROTOS on `lib` (a CDLL of the loader); returns {name: function}."""
    fns = {}
    for name, (res, *args) in PROTOS.items():
        f = getattr(lib, name)
        f.restype = res
        f.argtypes = list(args)
        fns[name] = f
    return fns


def run(out: str, loader: str, want: str = "any", spv_dir: str = SHADERS, size: int = common.SIZE,
        cdll=None, on_stage=None) -> dict:
    res = common.new_result("vk_tri", loader_arg=loader, want=want)
    res["env"] = {k: os.environ.get(k) for k in ("VK_ICD_FILENAMES", "VK_DRIVER_FILES", "LD_LIBRARY_PATH",
                                                 "VK_LOADER_DEBUG", "LP_NUM_THREADS", "XDG_DATA_HOME")}
    m = res["metrics"]
    m["vk"] = {}
    t0 = time.time()
    try:
        _run(out, loader, want, spv_dir, size, cdll or C.CDLL, res, on_stage)
    except _Stop:
        pass
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
        res["notes"].append(f"exception at stage {res['stage']}: {e!r}")
    m["probe_s"] = round(time.time() - t0, 3)
    m["mapped_gpu_libs"] = common.mapped_libs()
    return res


def _run(out, loader, want, spv_dir, size, cdll, res, on_stage=None):
    m, vkr = res["metrics"], res["metrics"]["vk"]

    def stage(name):
        res["stage"] = name
        if on_stage is not None:
            on_stage(res)

    def check(name, r):
        vkr[name] = vk_name(r)
        if int(r) != 0:
            res["notes"].append(f"{name} returned {vk_name(r)}")
            raise _Stop()

    stage("load_spv")
    vert, vert_n, m["vert_spv_sha256"] = load_spv(os.path.join(spv_dir, "tri.vert.spv"))
    frag, frag_n, m["frag_spv_sha256"] = load_spv(os.path.join(spv_dir, "tri.frag.spv"))
    stage("load_loader")
    try:
        lib = cdll(loader)
    except OSError as e:
        res["notes"].append(f"cannot load the Vulkan loader {loader}: {e!r}")
        raise _Stop()
    fn = bind_api(lib)
    m["loader_mapped"] = [p for p in common.mapped_libs() if "libvulkan.so" in os.path.basename(p)]
    ver = u32(0)
    if fn["vkEnumerateInstanceVersion"](C.byref(ver)) == 0:
        m["loader_instance_version"] = version_str(ver.value)

    stage("create_instance")
    app = AppInfo(sType=ST_APP, pApplicationName=b"vpd-vk-tri", applicationVersion=1, pEngineName=b"none",
                  engineVersion=1, apiVersion=API_1_1)
    ici = InstanceCI(sType=ST_INSTANCE, pApplicationInfo=C.pointer(app))
    inst = D()
    check("vkCreateInstance", fn["vkCreateInstance"](C.byref(ici), None, C.byref(inst)))

    stage("enumerate_devices")
    n = u32(0)
    check("vkEnumeratePhysicalDevices", fn["vkEnumeratePhysicalDevices"](inst, C.byref(n), None))
    pdevs = (D * max(n.value, 1))()
    check("vkEnumeratePhysicalDevices", fn["vkEnumeratePhysicalDevices"](inst, C.byref(n), pdevs))
    devices, chosen = [], None
    for i in range(n.value):
        raw = C.create_string_buffer(2048)
        fn["vkGetPhysicalDeviceProperties"](pdevs[i], raw)
        d = dict(parse_props(raw.raw), index=i)
        nq = u32(0)
        fn["vkGetPhysicalDeviceQueueFamilyProperties"](pdevs[i], C.byref(nq), None)
        qf = (QueueFamilyProps * max(nq.value, 1))()
        fn["vkGetPhysicalDeviceQueueFamilyProperties"](pdevs[i], C.byref(nq), qf)
        d["queue_families"] = [{"flags": qf[k].queueFlags, "count": qf[k].queueCount} for k in range(nq.value)]
        gq = [k for k in range(nq.value) if qf[k].queueFlags & QUEUE_GRAPHICS and qf[k].queueCount > 0]
        d["graphics_queue_family"] = gq[0] if gq else None
        devices.append(d)
        if chosen is None and gq and wants(d, want):
            chosen = d
    m["devices"] = devices
    if chosen is None:
        res["notes"].append(f"no physical device with a graphics queue matches want={want} ({len(devices)} found)")
        raise _Stop()
    m["device"] = {k: chosen[k] for k in ("index", "name", "type", "vendorID", "deviceID", "apiVersion")}
    pdev, qfam = pdevs[chosen["index"]], chosen["graphics_queue_family"]
    mem = MemProps()
    fn["vkGetPhysicalDeviceMemoryProperties"](pdev, C.byref(mem))

    stage("create_device")
    prio = (f32 * 1)(1.0)
    qci = DeviceQueueCI(sType=ST_DEVICE_QUEUE, queueFamilyIndex=qfam, queueCount=1, pQueuePriorities=ptr(prio, f32))
    dci = DeviceCI(sType=ST_DEVICE, queueCreateInfoCount=1, pQueueCreateInfos=C.pointer(qci))
    dev = D()
    check("vkCreateDevice", fn["vkCreateDevice"](pdev, C.byref(dci), None, C.byref(dev)))
    queue = D()
    fn["vkGetDeviceQueue"](dev, qfam, 0, C.byref(queue))

    stage("create_image")
    rng = SubresRange(aspectMask=ASPECT_COLOR, baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1)
    ici2 = ImageCI(sType=ST_IMAGE, imageType=IMAGE_TYPE_2D, format=FORMAT_R8G8B8A8_UNORM,
                   extent=Extent3D(size, size, 1), mipLevels=1, arrayLayers=1, samples=SAMPLE_1,
                   tiling=TILING_OPTIMAL, usage=USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC, sharingMode=0,
                   initialLayout=LAYOUT_UNDEFINED)
    image = H()
    check("vkCreateImage", fn["vkCreateImage"](dev, C.byref(ici2), None, C.byref(image)))
    req = MemReqs()
    fn["vkGetImageMemoryRequirements"](dev, image, C.byref(req))
    mt = find_memory_type(mem, req.memoryTypeBits, MEM_DEVICE_LOCAL)
    if mt is None:
        mt = find_memory_type(mem, req.memoryTypeBits, 0)
    m["image_memory_type"] = mt
    imem = H()
    check("vkAllocateMemory(image)", fn["vkAllocateMemory"](
        dev, C.byref(MemAllocInfo(sType=ST_MEM_ALLOC, allocationSize=req.size, memoryTypeIndex=mt)), None,
        C.byref(imem)))
    check("vkBindImageMemory", fn["vkBindImageMemory"](dev, image, imem, 0))
    view = H()
    check("vkCreateImageView", fn["vkCreateImageView"](dev, C.byref(ImageViewCI(
        sType=ST_IMAGE_VIEW, image=image.value, viewType=IMAGE_VIEW_TYPE_2D, format=FORMAT_R8G8B8A8_UNORM,
        components=ComponentMapping(0, 0, 0, 0), subresourceRange=rng)), None, C.byref(view)))

    stage("create_render_pass")
    att = AttachmentDesc(format=FORMAT_R8G8B8A8_UNORM, samples=SAMPLE_1, loadOp=LOAD_CLEAR, storeOp=STORE_STORE,
                         stencilLoadOp=LOAD_DONT_CARE, stencilStoreOp=STORE_DONT_CARE,
                         initialLayout=LAYOUT_UNDEFINED, finalLayout=LAYOUT_COLOR_ATTACHMENT)
    cref = AttachmentRef(attachment=0, layout=LAYOUT_COLOR_ATTACHMENT)
    sub = SubpassDesc(pipelineBindPoint=BIND_GRAPHICS, colorAttachmentCount=1, pColorAttachments=C.pointer(cref))
    rp = H()
    check("vkCreateRenderPass", fn["vkCreateRenderPass"](dev, C.byref(RenderPassCI(
        sType=ST_RENDER_PASS, attachmentCount=1, pAttachments=C.pointer(att), subpassCount=1,
        pSubpasses=C.pointer(sub))), None, C.byref(rp)))
    views = (H * 1)(view.value)
    fb = H()
    check("vkCreateFramebuffer", fn["vkCreateFramebuffer"](dev, C.byref(FramebufferCI(
        sType=ST_FRAMEBUFFER, renderPass=rp.value, attachmentCount=1, pAttachments=ptr(views, H), width=size,
        height=size, layers=1)), None, C.byref(fb)))

    stage("create_pipeline")
    vmod, fmod = H(), H()
    check("vkCreateShaderModule(vert)", fn["vkCreateShaderModule"](dev, C.byref(ShaderModuleCI(
        sType=ST_SHADER_MODULE, codeSize=vert_n, pCode=ptr(vert, u32))), None, C.byref(vmod)))
    check("vkCreateShaderModule(frag)", fn["vkCreateShaderModule"](dev, C.byref(ShaderModuleCI(
        sType=ST_SHADER_MODULE, codeSize=frag_n, pCode=ptr(frag, u32))), None, C.byref(fmod)))
    stages = (ShaderStageCI * 2)(
        ShaderStageCI(sType=ST_STAGE, stage=STAGE_VERTEX, module=vmod.value, pName=b"main"),
        ShaderStageCI(sType=ST_STAGE, stage=STAGE_FRAGMENT, module=fmod.value, pName=b"main"))
    vin = VertexInputCI(sType=ST_VERTEX_INPUT)
    ia = InputAssemblyCI(sType=ST_INPUT_ASSEMBLY, topology=TOPOLOGY_TRIANGLE_LIST)
    vpt = Viewport(0.0, 0.0, float(size), float(size), 0.0, 1.0)
    sc = Rect2D(Offset2D(0, 0), Extent2D(size, size))
    vps = ViewportCI(sType=ST_VIEWPORT, viewportCount=1, pViewports=C.pointer(vpt), scissorCount=1,
                     pScissors=C.pointer(sc))
    rs = RasterCI(sType=ST_RASTER, polygonMode=POLYGON_FILL, cullMode=CULL_NONE, frontFace=FRONT_CCW,
                  lineWidth=1.0)
    ms = MultisampleCI(sType=ST_MULTISAMPLE, rasterizationSamples=SAMPLE_1, minSampleShading=1.0)
    ba = BlendAttachment(blendEnable=0, colorWriteMask=COLOR_RGBA)
    bs = BlendCI(sType=ST_BLEND, attachmentCount=1, pAttachments=C.pointer(ba))
    layout = H()
    check("vkCreatePipelineLayout", fn["vkCreatePipelineLayout"](
        dev, C.byref(PipelineLayoutCI(sType=ST_PIPELINE_LAYOUT)), None, C.byref(layout)))
    gp = GraphicsPipelineCI(sType=ST_GRAPHICS_PIPELINE, stageCount=2, pStages=ptr(stages, ShaderStageCI),
                            pVertexInputState=C.pointer(vin), pInputAssemblyState=C.pointer(ia),
                            pViewportState=C.pointer(vps), pRasterizationState=C.pointer(rs),
                            pMultisampleState=C.pointer(ms), pColorBlendState=C.pointer(bs), layout=layout.value,
                            renderPass=rp.value, subpass=0, basePipelineIndex=-1)
    pipe = H()
    t = time.time()
    r = fn["vkCreateGraphicsPipelines"](dev, 0, 1, C.byref(gp), None, C.byref(pipe))
    m["pipeline_create_ms"] = round(1000 * (time.time() - t), 2)
    check("vkCreateGraphicsPipelines", r)

    stage("create_readback")
    nbytes = size * size * 4
    buf = H()
    check("vkCreateBuffer", fn["vkCreateBuffer"](dev, C.byref(BufferCI(
        sType=ST_BUFFER, size=nbytes, usage=BUF_USAGE_TRANSFER_DST, sharingMode=0)), None, C.byref(buf)))
    breq = MemReqs()
    fn["vkGetBufferMemoryRequirements"](dev, buf, C.byref(breq))
    bt = find_memory_type(mem, breq.memoryTypeBits, MEM_HOST_VISIBLE | MEM_HOST_COHERENT)
    if bt is None:
        res["notes"].append("no HOST_VISIBLE | HOST_COHERENT memory type for the readback buffer")
        raise _Stop()
    m["buffer_memory_type"] = bt
    bmem = H()
    check("vkAllocateMemory(buffer)", fn["vkAllocateMemory"](
        dev, C.byref(MemAllocInfo(sType=ST_MEM_ALLOC, allocationSize=breq.size, memoryTypeIndex=bt)), None,
        C.byref(bmem)))
    check("vkBindBufferMemory", fn["vkBindBufferMemory"](dev, buf, bmem, 0))

    stage("record")
    pool = H()
    check("vkCreateCommandPool", fn["vkCreateCommandPool"](
        dev, C.byref(CommandPoolCI(sType=ST_CMD_POOL, queueFamilyIndex=qfam)), None, C.byref(pool)))
    cmd = D()
    check("vkAllocateCommandBuffers", fn["vkAllocateCommandBuffers"](
        dev, C.byref(CommandBufferAI(sType=ST_CMD_ALLOC, commandPool=pool.value, level=0, commandBufferCount=1)),
        C.byref(cmd)))
    check("vkBeginCommandBuffer", fn["vkBeginCommandBuffer"](
        cmd, C.byref(CommandBufferBI(sType=ST_CMD_BEGIN, flags=CB_ONE_TIME))))
    clear = ClearValue((f32 * 4)(*common.CLEAR))
    fn["vkCmdBeginRenderPass"](cmd, C.byref(RenderPassBI(
        sType=ST_RP_BEGIN, renderPass=rp.value, framebuffer=fb.value,
        renderArea=Rect2D(Offset2D(0, 0), Extent2D(size, size)), clearValueCount=1,
        pClearValues=C.pointer(clear))), 0)
    fn["vkCmdBindPipeline"](cmd, BIND_GRAPHICS, pipe)
    fn["vkCmdDraw"](cmd, 3, 1, 0, 0)
    fn["vkCmdEndRenderPass"](cmd)
    ib = ImageBarrier(sType=ST_IMAGE_BARRIER, srcAccessMask=ACC_COLOR_WRITE, dstAccessMask=ACC_TRANSFER_READ,
                      oldLayout=LAYOUT_COLOR_ATTACHMENT, newLayout=LAYOUT_TRANSFER_SRC,
                      srcQueueFamilyIndex=QUEUE_FAMILY_IGNORED, dstQueueFamilyIndex=QUEUE_FAMILY_IGNORED,
                      image=image.value, subresourceRange=rng)
    fn["vkCmdPipelineBarrier"](cmd, PS_COLOR_OUTPUT, PS_TRANSFER, 0, 0, None, 0, None, 1, C.byref(ib))
    region = BufferImageCopy(bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                             imageSubresource=SubresLayers(ASPECT_COLOR, 0, 0, 1), imageOffset=Offset3D(0, 0, 0),
                             imageExtent=Extent3D(size, size, 1))
    fn["vkCmdCopyImageToBuffer"](cmd, image, LAYOUT_TRANSFER_SRC, buf, 1, C.byref(region))
    bb = BufferBarrier(sType=ST_BUFFER_BARRIER, srcAccessMask=ACC_TRANSFER_WRITE, dstAccessMask=ACC_HOST_READ,
                       srcQueueFamilyIndex=QUEUE_FAMILY_IGNORED, dstQueueFamilyIndex=QUEUE_FAMILY_IGNORED,
                       buffer=buf.value, offset=0, size=nbytes)
    fn["vkCmdPipelineBarrier"](cmd, PS_TRANSFER, PS_HOST, 0, 0, None, 1, C.byref(bb), 0, None)
    check("vkEndCommandBuffer", fn["vkEndCommandBuffer"](cmd))

    stage("submit")
    fence = H()
    check("vkCreateFence", fn["vkCreateFence"](dev, C.byref(FenceCI(sType=ST_FENCE)), None, C.byref(fence)))
    cmds = (D * 1)(cmd.value)
    si = SubmitInfo(sType=ST_SUBMIT, commandBufferCount=1, pCommandBuffers=ptr(cmds, D))
    t = time.time()
    check("vkQueueSubmit", fn["vkQueueSubmit"](queue, 1, C.byref(si), fence))
    fences = (H * 1)(fence.value)
    r = fn["vkWaitForFences"](dev, 1, fences, 1, FENCE_TIMEOUT_NS)
    m["submit_to_fence_ms"] = round(1000 * (time.time() - t), 2)
    check("vkWaitForFences", r)

    stage("readback")
    mapped = vp()
    check("vkMapMemory", fn["vkMapMemory"](dev, bmem, 0, nbytes, 0, C.byref(mapped)))
    data = C.string_at(mapped.value, nbytes)
    pv = common.pixel_verdict(data, size, size)
    m["pixels"] = pv
    err = common.save_rgba_png(os.path.join(out, "tri.png"), data, size, size)
    if err:
        res["notes"].append(f"png: {err}")
    res["pass"] = bool(pv["ok"])
    if not pv["ok"]:
        res["notes"].append(f"pixel rule failed: centre {pv['centre']}, corners {pv['corners']}"
                            f"{' (all zero)' if pv['all_zero'] else ''}")
    stage("done")
