<template>
  <div class="login-page">
    <div class="login-page__theme">
      <ThemeToggle />
    </div>

    <div class="login-card">
      <div class="login-card__brand">
        <span class="login-card__mark" aria-hidden="true">V</span>
        <h1 class="login-card__title">{{ appConfig.appName }}</h1>
        <p class="login-card__subtitle">企业知识库智能助手</p>
      </div>

      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        size="large"
        @submit.prevent="handleSubmit"
      >
        <el-form-item prop="username">
          <el-input
            v-model="form.username"
            :prefix-icon="User"
            placeholder="请输入用户名"
            autocomplete="username"
            clearable
          />
        </el-form-item>

        <el-form-item prop="password">
          <el-input
            v-model="form.password"
            :prefix-icon="Lock"
            type="password"
            placeholder="请输入密码"
            autocomplete="current-password"
            show-password
          />
        </el-form-item>

        <el-form-item>
          <el-button
            class="login-card__submit"
            type="primary"
            native-type="submit"
            :loading="loading"
          >
            登 录
          </el-button>
        </el-form-item>
      </el-form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Lock, User } from '@element-plus/icons-vue'
import type { FormInstance, FormRules } from 'element-plus'

import ThemeToggle from '@/components/common/ThemeToggle.vue'
import { appConfig } from '@/config'
import { useUserStore } from '@/stores/user'

interface LoginForm {
  username: string
  password: string
}

const formRef = ref<FormInstance>()
const form = reactive<LoginForm>({
  username: '',
  password: '',
})

const rules: FormRules<LoginForm> = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { max: 100, message: '用户名最长 100 位', trigger: 'blur' },
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 1, max: 128, message: '密码长度为 1-128 位', trigger: 'blur' },
  ],
}

const loading = ref(false)
const route = useRoute()
const router = useRouter()
const userStore = useUserStore()

/** 只接受站内路径，避免 redirect 被利用跳转到外部站点。 */
function resolveRedirect(): string {
  const redirect = route.query.redirect
  if (typeof redirect === 'string' && redirect.startsWith('/') && !redirect.startsWith('//')) {
    return redirect
  }
  return '/'
}

async function handleSubmit(): Promise<void> {
  if (loading.value || !formRef.value) return

  const valid = await formRef.value.validate().catch(() => false)
  if (!valid) return

  loading.value = true
  try {
    await userStore.login({ ...form })
    ElMessage.success('登录成功')
    await router.replace(resolveRedirect())
  } catch {
    // 错误提示由请求层统一处理
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding: var(--app-space-lg);
  background: var(--app-login-background);
}

.login-page__theme {
  position: absolute;
  top: var(--app-space-lg);
  right: var(--app-space-lg);
}

.login-card {
  width: 100%;
  max-width: var(--app-login-card-width);
  padding: var(--app-space-xl) var(--app-space-2xl);
  background-color: var(--app-bg-card);
  border: 1px solid var(--app-border-color-light);
  border-radius: var(--app-radius-lg);
  box-shadow: var(--app-shadow-card);
}

.login-card__brand {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: var(--app-space-xl);
}

.login-card__mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  margin-bottom: var(--app-space-sm);
  font-size: var(--app-font-size-lg);
  font-weight: 700;
  color: var(--app-text-on-primary);
  background: linear-gradient(135deg, var(--app-color-primary), var(--app-color-success));
  border-radius: var(--app-radius-lg);
}

.login-card__title {
  font-size: var(--app-font-size-2xl);
  font-weight: 600;
  color: var(--app-text-primary);
  letter-spacing: 1px;
}

.login-card__subtitle {
  margin-top: var(--app-space-sm);
  font-size: var(--app-font-size-base);
  color: var(--app-text-secondary);
}

.login-card__submit {
  width: 100%;
}
</style>
